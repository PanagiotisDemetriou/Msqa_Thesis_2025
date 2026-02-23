import os 
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, default_collate
from pointcept.utils.config import Config as PCConfig
from pointcept.datasets.transform import Compose
from pointcept.datasets.utils import collate_fn
from torch_scatter import scatter_mean

def pool_features_scatter(obj_data):
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
               emb = torch.zeros(embed_dim, device=obj_feat.device)
         else:
               emb = obj_feat.mean(dim=0)          # → (feat_dim,)
         
         scene_embeds.append(emb)
         scene_mask.append(True)
      
      # pad to fixed max_objects per scene if needed
      num_real = len(scene_embeds)
      if num_real == 0:
         scene_embeds_padded = torch.zeros(60, embed_dim, device=obj_feat.device)
         scene_mask_padded = torch.zeros(60, dtype=torch.bool, device=obj_feat.device)
      else:
         scene_embeds_padded = torch.stack(scene_embeds)                     # (num_real, embed_dim)
         scene_embeds_padded = torch.nn.functional.pad(
               scene_embeds_padded,
               (0, 0, 0, 60 - num_real),   # pad on object dimension
               value=0.0
         )
         scene_mask_padded = torch.nn.functional.pad(
               torch.tensor(scene_mask, device=obj_feat.device),
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
   def __init__(self, cfg):
      self.ptv3_cfg = ptv3_cfg = PCConfig.fromfile(cfg.model.prompter.model.vision.args.ptv3_cfg_path)
      self.transform_cfg = ptv3_cfg.data.train.datasets[1]['transform']
      self.transform = Compose(self.transform_cfg)

      self.condition = 'ScanNet'
      

   def create_data_dict(self, data):
      assert 'scene_fts' in data, "Expected 'scene_fts' in data"
      assert 'instance_ids' in data, "Expected 'instance_ids' in data"
      assert 'segments' in data, "Expected 'segments' in data"
      scene_fts = data['scene_fts']
      inst_labels = data['instance_ids']
      assert scene_fts.shape[1] == 9, f"Expected 9 features (coords, color, normals), got {scene_fts.shape[1]}"
      
      
      coord = scene_fts[:,:3].numpy()  
      color = scene_fts[:,3:6].numpy()
      normals = scene_fts[:,6:9].numpy()
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
     
   def prepare_data(self, data_dict):
      result_dict = self.transform(data_dict)
      return result_dict

   # XXXX # prepeiu na dw ti ginetai me ta scenes pou allazoun kai na sasw to pooling gia batches 
   # Problems:
   # 1. Instance IDs missing due to transformation (e.g., cropping) -> can't pool by instance
   # 2. Slow
   # 3 .Check Masks 
   

   def pool_object_features(self, data, obj_ids):
      inst_dct = {}
      unique_inst_ids = torch.unique(data['inst_id'])
      
      # Break points into objects based on their instance ids
      for i in unique_inst_ids:
         mask = data['inst_id'] == i     # time consuming
         inst_dct[int(i.item())] = data['feat'][mask]
         # obj_pcds.append(data['feat'][mask])                    
      # data['pooled_fts'] = obj_pcds
      #print(f"Number of objects: {len(obj_pcds)}")
      # print(f"Unique instance IDs: {unique_inst_ids}")
      # print(f"Number of objects: {len(inst_dct.keys())}")
      # print(f"Object IDs in inst_dct: {list(inst_dct.keys())}")
      # print(f"Assetions = {list(inst_dct.keys()) == unique_inst_ids.tolist()}")
 
      accumulation = 0
      obj_data = []  # will become: list of length B, where each element is a list of tensors

      for b, row in enumerate(obj_ids):           # row = obj_ids[b]  shape [60] or similar
         scene_objects = []                      # list of feature tensors for this scene
         
         valid_row = row[row >= 0]               # ignore padding values (assuming <0 = pad)
         if valid_row.numel() == 0:
            obj_data.append(scene_objects)      # empty scene
            accumulation += 1                   # or 0 — depends on your ID scheme
            continue

         seen = set()  # optional: avoid duplicate local ids if they appear multiple times

         for lid_tensor in row:
            lid = lid_tensor.item()
            if lid < 0 or lid in seen:
                  continue
            seen.add(lid)

            global_id = lid + accumulation

            if global_id in inst_dct:
                  feats = inst_dct[global_id]              # shape: [num_points_this_obj, feat_dim]
                  scene_objects.append(feats)

         obj_data.append(scene_objects)

         # Update accumulation *after* processing the whole scene
         max_in_scene = valid_row.max().item()
         accumulation += max_in_scene + 1

         # print(f"Batch {b}: {len(scene_objects)} objects collected | Accumulation now: {accumulation}")

      # ────────────────────────────────────────────────
      # Optional debug prints (safe now)
      # ────────────────────────────────────────────────
      # for b, scene_list in enumerate(obj_data):
      #    print(f"Scene {b}: {len(scene_list)} objects")
      #    if scene_list:
      #       print(f"  → first object shape: {scene_list[0].shape}")
      #       print(f"  → last  object shape: {scene_list[-1].shape}")
      #    else:
      #       print("  → no objects")
               


      obj_embeds, obj_mask = pool_features_scatter(obj_data)
      
      return obj_embeds, obj_mask

   
   

      
   
        