# # TO Run
# # PYTHONPATH="$PWD:$PWD/msr3d:$PWD/Pointcept_main:$PYTHONPATH" python data_transform.py
# import os
# import torch
# import sys
# from pathlib import Path
# import numpy as np
# from scipy import sparse
# from data.datasets.scannet_base import ScanNetBase
# from pointcept.models.utils import batch2offset
# from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
# from pointcept.utils.config import Config
# from pointcept.models import build_model
# from collections import OrderedDict
# import pointcept.utils.comm as comm
# from omegaconf import OmegaConf
# import torch_scatter
# import open3d as o3d

# def move_pointcept_data_to_device(data_dict, device):
#     """
#     Move all torch.Tensor values in a (possibly nested) dict/list to `device`.
#     Non-tensors (str, int, float, None, list of str, etc.) are left unchanged.
#     """
#     if isinstance(data_dict, torch.Tensor):
#         return data_dict.to(device, non_blocking=True)
#     if isinstance(data_dict, dict):
#         return {k: move_pointcept_data_to_device(v, device) for k, v in data_dict.items()}
#     if isinstance(data_dict, (list, tuple)):
#         return type(data_dict)(move_pointcept_data_to_device(v, device) for v in data_dict)
#     return data_dict
# def load_pointcept_checkpoint(model, weight_path, strict=False):
#     checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

#     sd = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

#     weight = OrderedDict()
#     for k, v in sd.items():
#         # normalize to "module." keys first (matches Pointcept logic)
#         if not k.startswith("module."):
#             k = "module." + k
#         # if single process, strip module.
#         if comm.get_world_size() == 1:
#             k = k[7:]
#         weight[k] = v

#     missing, unexpected = model.load_state_dict(weight, strict=strict)
#     print(f"Missing: {len(missing)}  Unexpected: {len(unexpected)}")
#     return model
# def transform_data(obj_pcds, obj_normals=None):
#    """Transform input data from Pointnet++ format to PTv3 format.
#       Args:
#          obj_pcds: (B, N, P, C) tensor, where B is batch size, N is number of objects,
#                      P is number of points per object, and C is the number of channels (e.g., 3 for xyz).
#          obj_normals: (B, N, P, 3) tensor of point normals if available.
#       Returns:
#          Point Object:
#                coord: (B*N*P, 3) tensor of point coordinates
#                feat: (B*N*P, C) tensor of point features (here C=3 for xyz)
#                batch: (B*N*P,) tensor indicating batch index for each point
#    """
#    batch_size, num_objs, num_points, num_channels = obj_pcds.size()
#    total_points = batch_size * num_objs * num_points
#    point = {}
#    # flattened coordinates and features
#    point['coord'] = obj_pcds[..., :3].reshape(total_points, 3)  # Assuming C=3 for xyz
#    feat_rgb = obj_pcds[..., 3:].reshape(total_points, num_channels - 3)
#    feat_rgb = feat_rgb / 255.0

#    if obj_normals is not None:
#         normals_flat = obj_normals.reshape(total_points, 3)
#         point["feat"] = torch.cat([feat_rgb, normals_flat], dim=1)
#    else:
#         point["feat"] = feat_rgb

#    # batch indices
#    batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
#    batch_indices = batch_indices.expand(batch_size, num_objs, num_points).reshape(total_points)
#    point['batch'] = batch_indices.to(torch.long)

#    # offset
#    point['offset'] = batch2offset(point['batch'])

#    # grid size 
#    point['grid_size'] = 0.02  

#    # Save object ids for each point
#    obj_id = torch.arange(batch_size * num_objs, dtype=torch.long).repeat_interleave(num_points)
#    point["obj_id"] = obj_id

#    return point

# def pool_point_features_to_objects(point_feats: torch.Tensor,
#                                   obj_id: torch.Tensor,
#                                   num_objs: int = None,
#                                   reduce: str = "mean") -> torch.Tensor:
#     """
#     Pool per-point features into per-object features.

#     Args:
#         point_feats: (N_total, F) float tensor of point features (e.g., output of PTv3).
#         obj_id: (N_total,) long tensor mapping each point -> global object id [0..num_objs-1].
#         num_objs: total number of objects (B*N). If None, inferred as obj_id.max()+1.
#         reduce: reduction method: "mean", "max", "sum", "min".

#     Returns:
#         obj_feats: (num_objs, F) pooled object features.
#     """
#     if point_feats.ndim != 2:
#         raise ValueError(f"point_feats must be (N, C). Got {tuple(point_feats.shape)}")
#     if obj_id.ndim != 1:
#         raise ValueError(f"obj_id must be (N,). Got {tuple(obj_id.shape)}")
#     if point_feats.shape[0] != obj_id.shape[0]:
#         raise ValueError(f"Mismatched N_total: point_feats has {point_feats.shape[0]} rows, obj_id has {obj_id.shape[0]}")

#     if obj_id.dtype != torch.long:
#         obj_id = obj_id.long()

#     if num_objs is None:
#         if obj_id.numel() == 0:
#             raise ValueError("obj_id is empty; cannot infer num_objs.")
#         num_objs = int(obj_id.max().item()) + 1

#     if reduce not in {"mean", "max", "sum", "min"}:
#         raise ValueError(f"Unsupported reduce='{reduce}'. Choose from mean/max/sum/min.")

#     obj_feats = torch_scatter.scatter(
#         src=point_feats,
#         index=obj_id,
#         dim=0,
#         dim_size=num_objs,
#         reduce=reduce,
#     )
#     return obj_feats

# # def estimate_normals_whole_scene(obj_pcds_list, k=30, orient=True):
# #     """
# #     Estimate normals on the ENTIRE scene point cloud, then map back to objects.
    
# #     Args:
# #         obj_pcds_list: list of (N_i, 6) arrays [xyz, rgb] for each object
# #         k: number of neighbors for normal estimation
# #         orient: whether to orient normals consistently
    
# #     Returns:
# #         obj_normals_list: list of (N_i, 3) normal arrays matching input objects
# #     """
# #     # Step 1: Concatenate all object point clouds into one scene
# #     all_xyz = []
# #     point_to_obj = []  # Track which object each point belongs to
# #     point_counts = []  # Track how many points per object
    
# #     for obj_idx, obj_pcd in enumerate(obj_pcds_list):
# #         xyz = obj_pcd[:, :3]  # Extract xyz coordinates
# #         all_xyz.append(xyz)
# #         point_to_obj.extend([obj_idx] * len(xyz))
# #         point_counts.append(len(xyz))
    
# #     # Concatenate into single point cloud
# #     scene_xyz = np.vstack(all_xyz)  # (N_total, 3)
    
# #     # Step 2: Estimate normals on the WHOLE SCENE
# #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_xyz))
# #     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    
# #     if orient:
# #         pcd.orient_normals_consistent_tangent_plane(k)
    
# #     scene_normals = np.asarray(pcd.normals).astype(np.float32)  # (N_total, 3)
    
# #     # Step 3: Split normals back to individual objects
# #     obj_normals_list = []
# #     start_idx = 0
# #     for count in point_counts:
# #         end_idx = start_idx + count
# #         obj_normals_list.append(scene_normals[start_idx:end_idx])
# #         start_idx = end_idx
    
# #     return obj_normals_list

# # def resample_normals_to_fixed_size(obj_xyz, obj_normals_full, target_points=1024):
# #     """
# #     After resampling points to fixed size, we need to transfer normals.
# #     This uses nearest neighbor to map normals from full point cloud to sampled points.
    
# #     Args:
# #         obj_xyz: (P_sampled, 3) - the sampled/resampled point coordinates
# #         obj_normals_full: (P_original, 3) - normals from the full object point cloud
# #         target_points: expected number of sampled points
    
# #     Returns:
# #         normals_sampled: (P_sampled, 3) - normals for the sampled points
# #     """
# #     # Build KDTree on original points to find nearest neighbors
# #     # This assumes obj_xyz came from resampling the original point cloud
# #     # You'll need to pass the original xyz as well - see main() for proper usage
# #     pass  # See main() for actual implementation

# def main():
#    # Example usage
#    path = "pcd_with_global_alignment/scene0000_00"
#    normals_path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals"
#    weight_path = "/mnt/d/Thesis/PTv3/model_best.pth"  # Path to Pointcept checkpoint
#    cfg = OmegaConf.load("msr3d/configs/data.yaml")
#    cfg_obj = OmegaConf.load("msr3d/configs/msr3d.yaml")
#    cfg_ptv3 = Config.fromfile("/home/panagiotis/msqa/Msqa_Thesis_2025/Pointcept_main/configs/scannet/semseg-pt-v3m1-1-ppt-extreme.py")
#    loader = ScanNetBase(cfg, split="train")
#    loader.num_points = 1024

#    scan_id_in = "scene0000_00" 
   
#    normals = torch.load(os.path.join(normals_path, f"{scan_id_in}.pth"), weights_only=False)  
#    obj_normals_list = normals["obj_normals_list"]   
   
#    scan_id, one_scan = loader._load_one_scan(
#       scan_id_in,
#       load_inst_info=True,
#       load_pc_info=True
#    )
#    print("Keys in one_scan:", one_scan.keys())
#    obj_pcds_list = one_scan["obj_pcds"]  # list of (N_i, 6) arrays BEFORE resampling
#    obj_labels = one_scan["inst_labels"]
   
#    assert len(obj_pcds_list) == len(obj_normals_list)
#    #print("Estimating normals on full scene...")
#    #obj_normals_list = estimate_normals_whole_scene(obj_pcds_list, k=30, orient=True)
   
#    # Now add normals to the point clouds before resampling
#    obj_pcds_with_normals = []
#    for obj_pcd, obj_normal in zip(obj_pcds_list, obj_normals_list):
#        # Concatenate xyz, rgb, normals: (N, 6) + (N, 3) = (N, 9)
#        obj_pcd_with_normal = np.hstack([obj_pcd, obj_normal])
#        obj_pcds_with_normals.append(obj_pcd_with_normal)
   
#    # Now do the resampling/processing (which will resample normals too)
#    obj_fts, obj_locs, obj_boxes, obj_labels = loader._obj_processing_post(
#       obj_pcds=obj_pcds_with_normals,  # Now (N, 9) instead of (N, 6)
#       obj_labels=obj_labels,
#       is_need_bbox=False,
#       rot_aug=True
#    )
   
#    # Extract resampled xyz, rgb, and normals
#    obj_xyz = obj_fts[..., :3]      # (N_obj, P=1024, 3)
#    obj_rgb = obj_fts[..., 3:6]     # (N_obj, P=1024, 3)
#    obj_normals = obj_fts[..., 6:9] # (N_obj, P=1024, 3)
   
#    # Reconstruct obj_pcds with xyz + rgb only (for compatibility)
#    obj_pcds = torch.cat([obj_xyz, obj_rgb], dim=-1).unsqueeze(0)  # (B=1, N, P, 6)
#    obj_normals = obj_normals.unsqueeze(0)  # (B=1, N, P, 3)
   
#    print(f"obj_pcds tensor: {obj_pcds.shape}")      # (1, N, 1024, 6)
#    print(f"obj_normals tensor: {obj_normals.shape}")  # (1, N, 1024, 3)
   
#    # Transform to PTv3 format
#    point_data = transform_data(obj_pcds, obj_normals=obj_normals)
#    point_data["offset"] = batch2offset(point_data["batch"])

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    point_data = move_pointcept_data_to_device(point_data, device)
#    point_data["condition"] = ["ScanNet"]

# #    print("coord:", point_data["coord"].shape)     # (N*P, 3)
# #    print("feat:", point_data["feat"].shape)       # (N*P, 6) - [rgb + normals]
# #    print("offset:", point_data["offset"].shape)
   
#    model = build_model(cfg_ptv3.model).cuda().eval()
#    model = load_pointcept_checkpoint(model, weight_path, strict=False)
   
# #    with torch.no_grad():
# #        point_out = model(point_data)
# #    print(type(point_out))
# #    print(point_out.keys())
# #    for k, v in point_out.items():
# #     if torch.is_tensor(v):
# #         print(k, v.shape, v.dtype, v.device)
# #     else:
# #         print(k, type(v))

#    device = torch.device("cuda:0")
#    core = model.module if hasattr(model, "module") else model
#    core = core.to(device).eval()

#    # move point_data tensors to GPU (coord/feat/batch/offset/grid_size/...)
   

#    # condition (if your wrapper needs it for other paths; backbone usually ignores it, but safe)
#    point_data["condition"] = ["ScanNet"]

#    with torch.no_grad():
#        point_out = core.backbone(point_data)   # usually returns a Point object
# #    print("backbone return type:", type(point_emb))
# #    print("has feat:", hasattr(point_emb, "feat"))
# #    print("per-point embedding:", point_emb.feat.shape, point_emb.feat.device, point_emb.feat.dtype)
# #    # Run through transformer
# #    transformer = PointTransformerV3(cls_mode=False, enable_flash=False)
# #    point_out = transformer(point_data)                  
   
   
#    object_feats = pool_point_features_to_objects(
#        point_feats=point_out.feat,
#        obj_id=point_data["obj_id"],
#        num_objs=obj_pcds.shape[1],  
#        reduce="mean"
#    )
#    print("Pooled object feats shape:", object_feats.shape)

#    # Now do the resampling/processing (which will resample normals too)
# #    obj_fts, obj_locs, obj_boxes, obj_labels = loader._obj_processing_post(
# #       obj_pcds=obj_pcds_with_normals,  # Now (N, 9) instead of (N, 6)
# #       obj_labels=obj_labels,
# #       is_need_bbox=False,
# #       rot_aug=True
# #    )



# #    # ===== CRITICAL CHECK: Did _obj_processing_post preserve all 9 channels? =====
# #    print(f"\n=== CHECKING _obj_processing_post OUTPUT ===")
# #    print(f"obj_fts shape: {obj_fts.shape}")
# #    print(f"Expected: (N_objects, 1024, 9) - [xyz(3) + rgb(3) + normals(3)]")
   
# #    if obj_fts.shape[-1] != 9:
# #        print(f"⚠️ WARNING: _obj_processing_post returned {obj_fts.shape[-1]} channels instead of 9!")
# #        print(f"   This means normals were NOT preserved during processing.")
# #        print(f"   You may need to modify _obj_processing_post or use a different approach.")
# #        # Fallback: estimate normals AFTER resampling (less accurate but better than nothing)
# #        print(f"   Falling back to per-object normal estimation...")
# #        obj_xyz = obj_fts[..., :3].detach().cpu().numpy()
# #        obj_normals = np.zeros((obj_xyz.shape[0], obj_xyz.shape[1], 3), dtype=np.float32)
# #        for i in range(obj_xyz.shape[0]):
# #            obj_normals[i] = estimate_normals_whole_scene(obj_xyz[i], k=30, orient=False)
# #        obj_normals = torch.from_numpy(obj_normals)
# #        obj_rgb = obj_fts[..., 3:6]
# #        obj_xyz = obj_fts[..., :3]
# #    else:
# #        print(f"✓ Good! _obj_processing_post preserved all 9 channels")
# #        # Extract resampled xyz, rgb, and normals
# #        obj_xyz = obj_fts[..., :3]      # (N_obj, P=1024, 3)
# #        obj_rgb = obj_fts[..., 3:6]     # (N_obj, P=1024, 3)
# #        obj_normals = obj_fts[..., 6:9] # (N_obj, P=1024, 3)
   
# #    print(f"=========================================\n")
   
# #    # Extract resampled xyz, rgb, and normals
# #     # Right after transform_data(), add:
# #    print(f"\nFeature check in point_data:")
# #    print(f"point_data['feat'] contains RGB + Normals")
# #    print(f"First point RGB: {point_data['feat'][0, :3]}")  # Should be 0-1 range
# #    print(f"First point Normal: {point_data['feat'][0, 3:6]}")  # Should be unit vector
# #    print(f"Normal magnitude: {torch.norm(point_data['feat'][0, 3:6]).item():.4f}")  # Should be ~1.0
# if __name__ == "__main__":
#     main()

# TO RUN (example)
# PYTHONPATH="$PWD:$PWD/msr3d:$PWD/Pointcept_main:$PYTHONPATH" python batch_test_pointcept.py

import os
import numpy as np
import torch
import torch_scatter
from collections import OrderedDict

from omegaconf import OmegaConf
from data.datasets.scannet_base import ScanNetBase

from pointcept.utils.config import Config
from pointcept.models import build_model
from pointcept.models.utils import batch2offset
import pointcept.utils.comm as comm


# -----------------------------
# Utilities
# -----------------------------
def move_pointcept_data_to_device(data_dict, device):
    """
    Move all torch.Tensor values in a (possibly nested) dict/list to `device`.
    Non-tensors are left unchanged.
    """
    if isinstance(data_dict, torch.Tensor):
        return data_dict.to(device, non_blocking=True)
    if isinstance(data_dict, dict):
        return {k: move_pointcept_data_to_device(v, device) for k, v in data_dict.items()}
    if isinstance(data_dict, (list, tuple)):
        return type(data_dict)(move_pointcept_data_to_device(v, device) for v in data_dict)
    return data_dict


def load_pointcept_checkpoint(model, weight_path, strict=False):
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    sd = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    weight = OrderedDict()
    for k, v in sd.items():
        # normalize to "module." keys first (matches Pointcept logic)
        if not k.startswith("module."):
            k = "module." + k
        # if single process, strip module.
        if comm.get_world_size() == 1:
            k = k[7:]
        weight[k] = v

    missing, unexpected = model.load_state_dict(weight, strict=strict)
    print(f"[Checkpoint] Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    return model


def transform_data(obj_pcds, obj_normals=None, grid_size=0.02):
    """
    obj_pcds: (B, N, P, 6) with [xyz, rgb]
    obj_normals: (B, N, P, 3)
    Returns point dict compatible with PTv3 backbone.
    """
    batch_size, num_objs, num_points, num_channels = obj_pcds.size()
    total_points = batch_size * num_objs * num_points

    point = {}
    point["coord"] = obj_pcds[..., :3].reshape(total_points, 3)

    feat_rgb = obj_pcds[..., 3:].reshape(total_points, num_channels - 3) / 255.0
    if obj_normals is not None:
        normals_flat = obj_normals.reshape(total_points, 3)
        point["feat"] = torch.cat([feat_rgb, normals_flat], dim=1)
    else:
        point["feat"] = feat_rgb

    batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
    batch_indices = batch_indices.expand(batch_size, num_objs, num_points).reshape(total_points)
    point["batch"] = batch_indices.to(torch.long)

    point["offset"] = batch2offset(point["batch"])
    point["grid_size"] = grid_size

    # global object id for each point (0..B*N-1)
    obj_id = torch.arange(batch_size * num_objs, dtype=torch.long).repeat_interleave(num_points)
    point["obj_id"] = obj_id

    return point


def pool_point_features_to_objects(point_feats: torch.Tensor,
                                  obj_id: torch.Tensor,
                                  num_objs: int,
                                  reduce: str = "mean") -> torch.Tensor:
    """
    point_feats: (N_total, F)
    obj_id: (N_total,) in [0..num_objs-1]
    """
    if point_feats.ndim != 2:
        raise ValueError(f"point_feats must be (N, C). Got {tuple(point_feats.shape)}")
    if obj_id.ndim != 1:
        raise ValueError(f"obj_id must be (N,). Got {tuple(obj_id.shape)}")
    if point_feats.shape[0] != obj_id.shape[0]:
        raise ValueError(f"Mismatched N_total: {point_feats.shape[0]} vs {obj_id.shape[0]}")
    if reduce not in {"mean", "max", "sum", "min"}:
        raise ValueError("reduce must be one of: mean/max/sum/min")

    if obj_id.dtype != torch.long:
        obj_id = obj_id.long()

    return torch_scatter.scatter(
        src=point_feats,
        index=obj_id,
        dim=0,
        dim_size=num_objs,
        reduce=reduce,
    )


# -----------------------------
# Scene helpers
# -----------------------------
def build_point_data_for_scene(loader, scan_id_in, normals_path, rot_aug=False):
    """
    Returns:
      point_data: dict with coord/feat/batch/offset/grid_size/obj_id
      num_objs: int
      obj_labels: torch.LongTensor of length num_objs
    """
    normals_ckpt = torch.load(
        os.path.join(normals_path, f"{scan_id_in}.pth"),
        map_location="cpu",
        weights_only=False
    )
    obj_normals_list = normals_ckpt["obj_normals_list"]

    _, one_scan = loader._load_one_scan(
        scan_id_in,
        load_inst_info=True,
        load_pc_info=True
    )

    obj_pcds_list = one_scan["obj_pcds"]         # list of (Ni, 6) [xyz, rgb]
    obj_labels = one_scan["inst_labels"]         # list/array len = num_objs

    assert len(obj_pcds_list) == len(obj_normals_list), \
        f"{scan_id_in}: obj_pcds_list({len(obj_pcds_list)}) != obj_normals_list({len(obj_normals_list)})"

    # Ensure normals are numpy arrays and concatenate to (Ni, 9)
    obj_pcds_with_normals = []
    for pcd, nrm in zip(obj_pcds_list, obj_normals_list):
        if torch.is_tensor(nrm):
            nrm = nrm.detach().cpu().numpy()
        obj_pcds_with_normals.append(np.hstack([pcd, nrm]).astype(np.float32))

    # IMPORTANT: your _obj_processing_post must rotate normals if rot_aug=True.
    obj_fts, _, _, obj_labels = loader._obj_processing_post(
        obj_pcds=obj_pcds_with_normals,
        obj_labels=obj_labels,
        is_need_bbox=False,
        rot_aug=rot_aug
    )

    # obj_fts: (N_obj, P, 9) [xyz, rgb, normals]
    obj_xyz = obj_fts[..., :3]
    obj_rgb = obj_fts[..., 3:6]
    obj_normals = obj_fts[..., 6:9]

    # Build tensors expected by transform_data (B=1)
    obj_pcds = torch.cat([obj_xyz, obj_rgb], dim=-1).unsqueeze(0)  # (1, N, P, 6)
    obj_normals = obj_normals.unsqueeze(0)                         # (1, N, P, 3)

    point_data = transform_data(obj_pcds, obj_normals=obj_normals, grid_size=0.02)
    point_data["offset"] = batch2offset(point_data["batch"])
    point_data["condition"] = ["ScanNet"]

    num_objs = int(obj_fts.shape[0])
    return point_data, num_objs, obj_labels


def merge_point_data(scene_point_datas, scene_num_objs, grid_size=0.02):
    """
    Merge multiple per-scene point_data dicts into a single batched point_data.

    Each scene point_data is assumed to have batch==0 for all points; we rewrite it to scene index.
    obj_id is shifted by cumulative object count.
    """
    coords, feats, batchs, obj_ids = [], [], [], []
    obj_base = 0

    for b, (pd, nobj) in enumerate(zip(scene_point_datas, scene_num_objs)):
        coords.append(pd["coord"])
        feats.append(pd["feat"])

        batchs.append(torch.full_like(pd["batch"], fill_value=b))
        obj_ids.append(pd["obj_id"] + obj_base)

        obj_base += int(nobj)

    out = {
        "coord": torch.cat(coords, dim=0),
        "feat": torch.cat(feats, dim=0),
        "batch": torch.cat(batchs, dim=0).long(),
        "obj_id": torch.cat(obj_ids, dim=0).long(),
        "grid_size": grid_size,
        "condition": ["ScanNet"],
    }
    out["offset"] = batch2offset(out["batch"])
    total_objs = obj_base
    return out, total_objs


# -----------------------------
# Main
# -----------------------------
def main():
    # ---- Paths / Config ----
    normals_path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals"
    weight_path = "/mnt/d/Thesis/PTv3/model_best.pth"
    cfg = OmegaConf.load("msr3d/configs/data.yaml")

    cfg_ptv3 = Config.fromfile(
        "/home/panagiotis/msqa/Msqa_Thesis_2025/Pointcept_main/configs/scannet/semseg-pt-v3m1-1-ppt-extreme.py"
    )

    # ---- Choose scenes to batch ----
    scan_ids = [
        "scene0000_00",
        "scene0001_00",
        # add more here
    ]

    # ---- Loader ----
    loader = ScanNetBase(cfg, split="train")
    loader.num_points = 1024

    # ---- Build per-scene point dicts ----
    scene_pds, scene_nobjs, scene_labels = [], [], []
    for sid in scan_ids:
        pd, nobj, labels = build_point_data_for_scene(
            loader=loader,
            scan_id_in=sid,
            normals_path=normals_path,
            rot_aug=False,   # recommended for test stability
        )
        scene_pds.append(pd)
        scene_nobjs.append(nobj)
        scene_labels.append(labels)
        print(f"[Scene] {sid}: num_objs={nobj}")

    # ---- Merge into a single batch ----
    point_data, total_objs = merge_point_data(scene_pds, scene_nobjs, grid_size=0.02)
    print(f"[Batch] scenes={len(scan_ids)} total_objs={total_objs}")
    assert point_data["offset"].numel() == len(scan_ids), \
        f"offset size {point_data['offset'].numel()} != num scenes {len(scan_ids)}"

    # ---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg_ptv3.model).to(device).eval()
    model = load_pointcept_checkpoint(model, weight_path, strict=False)

    core = model.module if hasattr(model, "module") else model
    core = core.to(device).eval()

    # ---- Move input to device ----
    point_data = move_pointcept_data_to_device(point_data, device)

    # ---- Forward ----
    with torch.no_grad():
        point_out = core.backbone(point_data)  # expects dict-like Point input

    # ---- Pool per-point -> per-object ----
    object_feats = pool_point_features_to_objects(
        point_feats=point_out.feat,
        obj_id=point_data["obj_id"],
        num_objs=total_objs,
        reduce="mean"
    )

    print("[Output] pooled object feats:", tuple(object_feats.shape))
    assert object_feats.shape[0] == sum(scene_nobjs), \
        "Pooled object count does not match sum of per-scene object counts."

    # Optional: split back per-scene
    starts = np.cumsum([0] + scene_nobjs[:-1]).tolist()
    for sid, start, nobj in zip(scan_ids, starts, scene_nobjs):
        feats_scene = object_feats[start:start + nobj]
        print(f"[Split] {sid}: feats={tuple(feats_scene.shape)}")


if __name__ == "__main__":
    main()
