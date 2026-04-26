import os 
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, default_collate
from pointcept.utils.config import Config as PCConfig
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import collate_fn
from torch_scatter import scatter_mean

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
      # MSR3D stores colors in [-1, 1], while Pointcept's NormalizeColor
      # expects raw 0..255 RGB and divides by 255 internally.
      color = ((scene_fts[:,3:6].detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255)
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

      #   print(
      #       f"[Scene {b}] matched {matched_slots}/{valid_slots} selected objects after PTv3 preprocessing"
      #   )

    return obj_embeds, obj_mask
   
   

      
   
        
