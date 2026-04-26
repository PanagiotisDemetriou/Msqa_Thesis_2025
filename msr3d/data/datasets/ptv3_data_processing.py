import os 
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, default_collate
from pointcept.utils.config import Config as PCConfig
from pointcept.datasets.transform import Compose, ElasticDistortion
from pointcept.datasets.utils import collate_fn
from torch_scatter import scatter_mean
from scipy.spatial.transform import Rotation as R

def pool_features_scatter(obj_data,device):
   embed_dim = 64   # ← or 128, 256, ... — decide what you want as final object size

   # Option A: keep same feature dim (just average)
   pooled_objects = []
   masks = []  # per scene: which slots are real

   for scene_objs in obj_data:
      scene_embeds = []
      scene_mask = []
      
      for obj_feat in scene_objs:
         if obj_feat.shape[0] == 0:
               # rare edge case - empty object
               emb = torch.zeros(embed_dim, device=device)
         else:
               emb = obj_feat.mean(dim=0)          # → (feat_dim,)
         
         scene_embeds.append(emb)
         scene_mask.append(True)
      
      # pad to fixed max_objects per scene if needed
      num_real = len(scene_embeds)
      if num_real == 0:
         scene_embeds_padded = torch.zeros(60, embed_dim, device=device)
         scene_mask_padded = torch.zeros(60, dtype=torch.bool, device=device)
      else:
         scene_embeds_padded = torch.stack(scene_embeds)                     # (num_real, embed_dim)
         scene_embeds_padded = torch.nn.functional.pad(
               scene_embeds_padded,
               (0, 0, 0, 60 - num_real),   # pad on object dimension
               value=0.0
         )
         scene_mask_padded = torch.nn.functional.pad(
               torch.tensor(scene_mask, device=device),
               (0, 60 - num_real),
               value=False
         )
      
      pooled_objects.append(scene_embeds_padded)
      masks.append(scene_mask_padded)

   # Final batched tensors
   pooled_embeds = torch.stack(pooled_objects)     # (B, 60, embed_dim)
   valid_mask    = torch.stack(masks)              # (B, 60) bool

   return pooled_embeds, valid_mask

class PTV3DataProcessing():
   def __init__(self, cfg, deterministic_transform=False):
      #self.ptv3_cfg = ptv3_cfg = PCConfig.fromfile(cfg.args.ptv3_cfg_path)
      self.ptv3_cfg = ptv3_cfg = PCConfig.fromfile(cfg.model.prompter.model.vision.args.ptv3_cfg_path)
      self.deterministic_transform = deterministic_transform

      self.train_transform_cfg = ptv3_cfg.data.train.datasets[1]['transform']
      self.train_transform = Compose(self.train_transform_cfg)

      self.val_transform_cfg = ptv3_cfg.data.val['transform']
      self.val_transform = Compose(self.val_transform_cfg)

      self.condition = 'ScanNet'
      

   def create_data_dict(self, data):
      assert 'scene_fts' in data, "Expected 'scene_fts' in data"
      assert 'instance_ids' in data, "Expected 'instance_ids' in data"
      assert 'segments' in data, "Expected 'segments' in data"
      scene_fts = data['scene_fts']
      inst_labels = data['instance_ids']
      assert scene_fts.shape[1] == 9, f"Expected 9 features (coords, color, normals), got {scene_fts.shape[1]}"
      
      
      # coord = scene_fts[:,:3].numpy()  
      # color = scene_fts[:,3:6].numpy()
      # normals = scene_fts[:,6:9].numpy()

      coord = scene_fts[:,:3].detach().cpu().numpy()
      color = scene_fts[:,3:6].detach().cpu().numpy()
      normals = scene_fts[:,6:9].detach().cpu().numpy()
      condition = self.condition
      segments = data['segments']

      data_dict ={
         'coord': coord,
         'color': color,
         'normal': normals,
         'condition': condition, 
         'name': data['scan_id'],
         'inst_id' : inst_labels,
         'segment': segments,
      }
      return data_dict

   def collate_pointcloud(self, batch):
      collated_batch = collate_fn(batch)
      return collated_batch
     
   def prepare_data(self, data_dict, mode):
      if mode == "train" and not self.deterministic_transform:
         #print("doing training ----------")
         result_dict = self.train_transform(data_dict)
      else:
         #print("doing inference ----------")
         result_dict = self.val_transform(data_dict)
      return result_dict

   @staticmethod
   def _cfg_get(cfg, key, default=None):
      if isinstance(cfg, dict):
         return cfg.get(key, default)
      return getattr(cfg, key, default)

   @staticmethod
   def _fnv_hash_vec(arr):
      assert arr.ndim == 2
      arr = arr.astype(np.uint64, copy=False)
      hashed_arr = np.uint64(14695981039346656037) * np.ones(
         arr.shape[0], dtype=np.uint64
      )
      for j in range(arr.shape[1]):
         hashed_arr *= np.uint64(1099511628211)
         hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
      return hashed_arr

   @staticmethod
   def _ravel_hash_vec(arr):
      assert arr.ndim == 2
      arr = arr.copy()
      arr -= arr.min(0)
      arr = arr.astype(np.uint64, copy=False)
      arr_max = arr.max(0).astype(np.uint64) + 1

      keys = np.zeros(arr.shape[0], dtype=np.uint64)
      for j in range(arr.shape[1] - 1):
         keys += arr[:, j]
         keys *= arr_max[j + 1]
      keys += arr[:, -1]
      return keys

   @staticmethod
   def _yaw_quaternion_from_vector(vec):
      vec = np.asarray(vec, dtype=np.float32)
      vec[2] = 0.0
      norm = np.linalg.norm(vec[:2])
      if norm < 1e-6:
         return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
      vec[:2] /= norm
      angle = np.arctan2(vec[1], vec[0])
      return R.from_euler('xyz', [0.0, 0.0, angle]).as_quat().astype(np.float32)

   def _apply_transform_replay(
      self,
      scene_coord,
      scene_inst,
      anchor_loc,
      anchor_ori,
      selected_obj_ids,
      current_obj_locs,
      scene_index,
      mode,
   ):
      coord = np.asarray(scene_coord, dtype=np.float32).copy()
      inst = np.asarray(scene_inst).copy()
      anchor_loc = np.asarray(anchor_loc, dtype=np.float32).copy()
      anchor_ori = np.asarray(anchor_ori, dtype=np.float32).copy()
      selected_obj_ids = np.asarray(selected_obj_ids).copy()
      transformed_obj_locs = np.asarray(current_obj_locs, dtype=np.float32).copy()

      transform_cfg = self.train_transform_cfg if mode == "train" and not self.deterministic_transform else self.val_transform_cfg
      hash_fn_map = {"fnv": self._fnv_hash_vec, "ravel": self._ravel_hash_vec}

      for transform in transform_cfg:
         t_type = self._cfg_get(transform, "type")

         if t_type == "CenterShift":
            if coord.shape[0] == 0:
               continue
            apply_z = self._cfg_get(transform, "apply_z", True)
            x_min, y_min, z_min = coord.min(axis=0)
            x_max, y_max, _ = coord.max(axis=0)
            shift = np.array(
               [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min if apply_z else 0.0],
               dtype=np.float32,
            )
            coord -= shift
            anchor_loc -= shift
         elif t_type == "RandomDropout":
            if coord.shape[0] == 0:
               continue
            dropout_ratio = self._cfg_get(transform, "dropout_ratio", 0.2)
            dropout_application_ratio = self._cfg_get(transform, "dropout_application_ratio", 0.5)
            if random.random() < dropout_application_ratio:
               keep = int(coord.shape[0] * (1 - dropout_ratio))
               idx = np.random.choice(coord.shape[0], keep, replace=False)
               coord = coord[idx]
               inst = inst[idx]
         elif t_type == "RandomRotate":
            if random.random() > self._cfg_get(transform, "p", 0.5):
               continue
            angle_range = self._cfg_get(transform, "angle", [-1, 1])
            angle = np.random.uniform(angle_range[0], angle_range[1]) * np.pi
            rot_cos, rot_sin = np.cos(angle), np.sin(angle)
            axis = self._cfg_get(transform, "axis", "z")
            if axis == "x":
               rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]], dtype=np.float32)
            elif axis == "y":
               rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]], dtype=np.float32)
            elif axis == "z":
               rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]], dtype=np.float32)
            else:
               continue

            center = self._cfg_get(transform, "center", None)
            if center is None:
               if coord.shape[0] == 0:
                  center = np.zeros(3, dtype=np.float32)
               else:
                  x_min, y_min, z_min = coord.min(axis=0)
                  x_max, y_max, z_max = coord.max(axis=0)
                  center = np.array(
                     [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2],
                     dtype=np.float32,
                  )
            else:
               center = np.asarray(center, dtype=np.float32)

            coord = np.dot(coord - center, rot_t.T) + center
            anchor_loc = np.dot(anchor_loc - center, rot_t.T) + center
            if axis == "z":
               anchor_ori = (R.from_matrix(rot_t) * R.from_quat(anchor_ori)).as_quat().astype(np.float32)
         elif t_type == "RandomScale":
            scale_cfg = self._cfg_get(transform, "scale", [0.95, 1.05])
            anisotropic = self._cfg_get(transform, "anisotropic", False)
            scale = np.random.uniform(scale_cfg[0], scale_cfg[1], 3 if anisotropic else 1).astype(np.float32)
            coord *= scale
            anchor_loc *= scale
         elif t_type == "RandomFlip":
            flip_x = np.random.rand() < self._cfg_get(transform, "p", 0.5)
            flip_y = np.random.rand() < self._cfg_get(transform, "p", 0.5)
            if flip_x:
               coord[:, 0] = -coord[:, 0]
               anchor_loc[0] = -anchor_loc[0]
            if flip_y:
               coord[:, 1] = -coord[:, 1]
               anchor_loc[1] = -anchor_loc[1]
            if flip_x or flip_y:
               forward = R.from_quat(anchor_ori).apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
               if flip_x:
                  forward[0] = -forward[0]
               if flip_y:
                  forward[1] = -forward[1]
               anchor_ori = self._yaw_quaternion_from_vector(forward)
         elif t_type == "RandomJitter":
            if coord.shape[0] == 0:
               continue
            sigma = self._cfg_get(transform, "sigma", 0.01)
            clip = self._cfg_get(transform, "clip", 0.05)
            jitter = np.clip(sigma * np.random.randn(coord.shape[0], 3), -clip, clip).astype(np.float32)
            coord += jitter
         elif t_type == "ElasticDistortion":
            if coord.shape[0] == 0:
               continue
            distortion_params = self._cfg_get(transform, "distortion_params", None)
            if distortion_params is not None and random.random() < 0.95:
               for granularity, magnitude in distortion_params:
                  coord = ElasticDistortion.elastic_distortion(coord, granularity, magnitude)
         elif t_type == "GridSample":
            if coord.shape[0] == 0:
               continue
            grid_size = self._cfg_get(transform, "grid_size", 0.05)
            hash_type = self._cfg_get(transform, "hash_type", "fnv")
            sample_mode = self._cfg_get(transform, "mode", "train")
            scaled_coord = coord / np.array(grid_size)
            grid_coord = np.floor(scaled_coord).astype(int)
            min_coord = grid_coord.min(0)
            grid_coord -= min_coord
            key = hash_fn_map[hash_type](grid_coord)
            idx_sort = np.argsort(key)
            key_sort = key[idx_sort]
            _, _, count = np.unique(key_sort, return_inverse=True, return_counts=True)
            if sample_mode == "train":
               idx_select = (
                  np.cumsum(np.insert(count, 0, 0)[:-1])
                  + np.random.randint(0, count.max(), count.size) % count
               )
               idx_unique = idx_sort[idx_select]
               coord = coord[idx_unique]
               inst = inst[idx_unique]
         elif t_type == "SphereCrop":
            if coord.shape[0] == 0:
               continue
            sample_rate = self._cfg_get(transform, "sample_rate", None)
            point_max = int(sample_rate * coord.shape[0]) if sample_rate is not None else self._cfg_get(transform, "point_max", 80000)
            if coord.shape[0] > point_max:
               crop_mode = self._cfg_get(transform, "mode", "random")
               if crop_mode == "random":
                  center = coord[np.random.randint(coord.shape[0])]
               elif crop_mode == "center":
                  center = coord[coord.shape[0] // 2]
               else:
                  continue
               idx_crop = np.argsort(np.sum(np.square(coord - center), 1))[:point_max]
               coord = coord[idx_crop]
               inst = inst[idx_crop]
         elif t_type == "ShufflePoint":
            if coord.shape[0] == 0:
               continue
            shuffle_index = np.arange(coord.shape[0])
            np.random.shuffle(shuffle_index)
            coord = coord[shuffle_index]
            inst = inst[shuffle_index]

      scene_instance_base = scene_index * 100000
      for slot, lid in enumerate(selected_obj_ids):
         lid = int(lid)
         if lid < 0:
            continue
         mask = inst == (scene_instance_base + lid)
         if not np.any(mask):
            continue
         obj_points = coord[mask]
         transformed_obj_locs[slot, :3] = obj_points.mean(0)
         transformed_obj_locs[slot, 3:] = obj_points.max(0) - obj_points.min(0)

      return anchor_loc.astype(np.float32), anchor_ori.astype(np.float32), transformed_obj_locs.astype(np.float32)

   def transform_spatial_metadata(
      self,
      scene_coord,
      scene_inst,
      anchor_loc,
      anchor_ori,
      selected_obj_ids,
      current_obj_locs,
      scene_index,
      mode,
      py_state,
      np_state,
   ):
      current_py_state = random.getstate()
      current_np_state = np.random.get_state()
      try:
         random.setstate(py_state)
         np.random.set_state(np_state)
         return self._apply_transform_replay(
            scene_coord=scene_coord,
            scene_inst=scene_inst,
            anchor_loc=anchor_loc,
            anchor_ori=anchor_ori,
            selected_obj_ids=selected_obj_ids,
            current_obj_locs=current_obj_locs,
            scene_index=scene_index,
            mode=mode,
         )
      finally:
         random.setstate(current_py_state)
         np.random.set_state(current_np_state)

   # XXXX # 
   # def pool_object_features(self, data, obj_ids):
   #    inst_dct = {}
   #    unique_inst_ids = torch.unique(data['inst_id'])
      
   #    # Break points into objects based on their instance ids
   #    for i in unique_inst_ids:
   #       mask = data['inst_id'] == i     # time consuming
   #       inst_dct[int(i.item())] = data['feat'][mask]
   #       # obj_pcds.append(data['feat'][mask])                    
   #    # data['pooled_fts'] = obj_pcds
   #    #print(f"Number of objects: {len(obj_pcds)}")
   #    # print(f"Unique instance IDs: {unique_inst_ids}")
   #    # print(f"Number of objects: {len(inst_dct.keys())}")
   #    # print(f"Object IDs in inst_dct: {list(inst_dct.keys())}")
   #    # print(f"Assetions = {list(inst_dct.keys()) == unique_inst_ids.tolist()}")
 
   #    # Keep the objects that are selected for the other 3D Backbone
   #    accumulation = 0
   #    obj_data = []  

   #    for b, row in enumerate(obj_ids):           
   #       scene_objects = []                      
         
   #       valid_row = row[row >= 0]               
   #       if valid_row.numel() == 0:
   #          obj_data.append(scene_objects)      
   #          accumulation += 1                  
   #          continue

   #       seen = set()  

   #       for lid_tensor in row:
   #          lid = lid_tensor.item()
   #          if lid < 0 or lid in seen:
   #                continue
   #          seen.add(lid)

   #          global_id = lid + accumulation

   #          if global_id in inst_dct:
   #                feats = inst_dct[global_id]              
   #                scene_objects.append(feats)

   #       obj_data.append(scene_objects)

         
   #       max_in_scene = valid_row.max().item()
   #       accumulation += max_in_scene + 1

         

   #    # ────────────────────────────────────────────────
   #    # Optional debug prints (safe now)
   #    # ────────────────────────────────────────────────
   #    # for b, scene_list in enumerate(obj_data):
   #    #    print(f"Scene {b}: {len(scene_list)} objects")
   #    #    if scene_list:
   #    #       print(f"  → first object shape: {scene_list[0].shape}")
   #    #       print(f"  → last  object shape: {scene_list[-1].shape}")
   #    #    else:
   #    #       print("  → no objects")
               

   #    # Pool Point features to per object features
   #    obj_embeds, obj_mask = pool_features_scatter(obj_data)
      
   #    return obj_embeds, obj_mask
   # def pool_object_features(self, data, obj_ids):
   #    K = 100000
   #    max_objects = 60

   #    inst_dct = {}
   #    unique_inst_ids = torch.unique(data['inst_id'])

   #    # Build dictionary: encoded instance id -> point features
   #    for i in unique_inst_ids:
   #       iid = int(i.item())
   #       if iid < 0:   # skip ignore ids like -100
   #             continue
   #       mask = data['inst_id'] == i
   #       inst_dct[iid] = data['feat'][mask]

   #    obj_data = []
   #    kept_ids_per_scene = []   # ← ADD THIS ### 
   #    for b, row in enumerate(obj_ids):
   #       scene_objects = []
   #       valid_row = row[row >= 0]
   #       kept_ids = []         # ← track kept IDs ###
   #       # print(f"\n[scene {b}] row={row.tolist()}")
   #       # print(f"[scene {b}] valid_row={valid_row.tolist()}")

   #       seen = set()

   #       # 1. Add requested objects first, preserving order
   #       for lid_tensor in row:
   #             lid = int(lid_tensor.item())
   #             if lid < 0 or lid in seen:
   #                continue
   #             seen.add(lid)

   #             global_id = b * K + lid
   #             # print(f"scene={b}, local_id={lid}, global_id={global_id}, exists={global_id in inst_dct}")

   #             if global_id in inst_dct:
   #                scene_objects.append(inst_dct[global_id])

   #             if len(scene_objects) >= max_objects:
   #                break

   #       # 2. Fill remaining slots with other objects from the same scene
   #       if len(scene_objects) < max_objects:
   #             scene_start = b * K
   #             scene_end = (b + 1) * K

   #             # all encoded ids that belong to this scene
   #             scene_global_ids = sorted(
   #                gid for gid in inst_dct.keys()
   #                if scene_start <= gid < scene_end
   #             )

   #             for global_id in scene_global_ids:
   #                local_id = global_id - scene_start

   #                if len(scene_objects) >= max_objects:
   #                   break

   #                if local_id in seen:
   #                   continue

   #                scene_objects.append(inst_dct[global_id])
   #                kept_ids.append(local_id)   # ← record it ###
   #                seen.add(local_id)
   #       if len(scene_objects) == 0:
   #             print(f"  row={row.tolist()}")

   #       obj_data.append(scene_objects)
   #       kept_ids_per_scene.append(kept_ids)   # ← store per scene ### 
   #       print(f"OBJ_IDS: {obj_ids}")
   #       print(f"[Scene {b}] Kept object IDs ({len(kept_ids)}): {kept_ids}") ### 
   #    device = data['feat'].device
   #    obj_embeds, obj_mask = pool_features_scatter(obj_data, device = device)
   #    return obj_embeds, obj_mask
   def pool_object_features(self, data, obj_ids):
    K = 100000
    device = data['feat'].device

    if 'offset' not in data:
        raise ValueError("pool_object_features requires 'offset' in data")

    offset = data['offset'].cpu()
    feat = data['feat']
    inst_id = data['inst_id']

    assert offset[-1] == feat.shape[0], "Offset doesn't match feat length"

    batch_size = len(obj_ids)
    max_objects = obj_ids.shape[1]
    feat_dim = feat.shape[1]

    obj_embeds = torch.zeros(batch_size, max_objects, feat_dim, device=device, dtype=feat.dtype)
    obj_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool, device=device)

    for b in range(batch_size):
        start = offset[b].item()
        end = offset[b + 1].item()

        scene_feat = feat[start:end]
        scene_inst = inst_id[start:end]

        global_inst_dct = {}
        unique_globals = torch.unique(scene_inst)
        for g in unique_globals:
            g_int = int(g.item())
            if g_int < 0:
                continue
            global_inst_dct[g_int] = scene_feat[scene_inst == g]

        row = obj_ids[b]
        valid_slots = 0
        matched_slots = 0

        for slot, lid_t in enumerate(row):
            lid = int(lid_t.item())
            if lid < 0:
                continue

            valid_slots += 1
            global_id = b * K + lid
            obj_points = global_inst_dct.get(global_id)
            if obj_points is None or obj_points.numel() == 0:
                continue

            obj_embeds[b, slot] = obj_points.mean(dim=0)
            obj_mask[b, slot] = True
            matched_slots += 1

        print(
            f"[Scene {b}] matched {matched_slots}/{valid_slots} selected objects after PTv3 preprocessing"
        )

    return obj_embeds, obj_mask
   
   

      
   
        
