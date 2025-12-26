# TO Run
# PYTHONPATH="$PWD:$PWD/msr3d:$PWD/Pointcept_main:$PYTHONPATH" python data_transform.py
import os
import torch
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
from data.datasets.scannet_base import ScanNetBase
from pointcept.models.utils import batch2offset
from omegaconf import OmegaConf
def transform_data(obj_pcds):
   """Transform input data from Pointnet++ format to PTv3 format.
      Args:
         obj_pcds: (B, N, P, C) tensor, where B is batch size, N is number of objects,
                     P is number of points per object, and C is the number of channels (e.g., 3 for xyz).
      Returns:
         Point Object:
               coord: (B*N*P, 3) tensor of point coordinates
               feat: (B*N*P, C) tensor of point features (here C=3 for xyz)
               batch: (B*N*P,) tensor indicating batch index for each point
   """
   batch_size, num_objs, num_points, num_channels = obj_pcds.size()
   total_points = batch_size * num_objs * num_points
   point = {}
   # flattened coordinates and features
   point['coord'] = obj_pcds[..., :3].reshape(total_points, 3)  # Assuming C=3 for xyz
   point['feat'] = obj_pcds.reshape(total_points, num_channels)
   # check for normals for other datasets 
   # batch indices
   # (B, 1, 1) -> (B, 1) -> (B, 1, 1)
   batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
   # expand to (B, N, P) and then reshape to (B*N*P,)
   batch_indices = batch_indices.expand(batch_size, num_objs, num_points).reshape(total_points)
   point['batch'] = batch_indices.to(torch.long)

   # offset
   point['offset'] = batch2offset(point['batch'])

   # grid size 
   point['grid_size'] = 0.02  

   return point



def main():
   # Example usage
   path = "pcd_with_global_alignment/scene0000_00"
   cfg = OmegaConf.load("msr3d/configs/data.yaml")

   loader = ScanNetBase(cfg, split="train")
   loader.num_points = 1024  # required by _obj_processing_post

   scan_id_in = "scene0000_00"  # keep your current approach

   scan_id, one_scan = loader._load_one_scan(
      scan_id_in,
      load_inst_info=True,
      load_pc_info=True
   )

   # one_scan is a dict; get list of per-object point clouds
   obj_pcds_list = one_scan["obj_pcds"]          # list of (Ni, 6) arrays
   obj_labels = one_scan["inst_labels"]          # list/array of labels

   # convert list -> fixed tensor (N, P, C)
   obj_fts, obj_locs, obj_boxes, obj_labels = loader._obj_processing_post(
      obj_pcds=obj_pcds_list,
      obj_labels=obj_labels,
      is_need_bbox=False,
      rot_aug=True
   )  # obj_fts: (N, P, C)

   obj_pcds = obj_fts.unsqueeze(0)  # (B=1, N, P, C)

   point_data = transform_data(obj_pcds)
   point_data["offset"] = batch2offset(point_data["batch"])

   print("obj_pcds tensor:", obj_pcds.shape)      # (1, N, P, C)
   print("coord:", point_data["coord"].shape)     # (N*P, 3)
   print("feat:", point_data["feat"].shape)       # (N*P, C)
   print("offset:", point_data["offset"].shape)
   # point_data = transform_data(obj_pcds)
   print(obj_pcds)
   print("Point Coordinates Shape:", point_data['offset'])  # Should be (B*N*P, 3)
   torch.save(obj_pcds, "new_data.pth")
if __name__ == "__main__":
    main()