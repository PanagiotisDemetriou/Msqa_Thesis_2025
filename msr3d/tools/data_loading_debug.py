#!/usr/bin/env python3
"""
debug_scannet_scene_and_ptv3_batching.py

1) Loads ScanNet scene via ScanNetBase._load_one_scan() (global alignment + normals cache)
2) Prints core scene/object stats
3) Uses YOUR actual transform_obj_pcds_to_pointcept() implementation (imported)
4) Prints Pointcept batching details (batch ids, counts, offsets, obj_id)

Usage:
  python debug_scannet_scene_and_ptv3_batching.py \
    --cfg msr3d/configs/data.yaml --split train --scan_id scene0000_00 \
    --num_points 1024 --grid_size 0.02
"""

import argparse
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import pandas as pd 
import torch
from omegaconf import OmegaConf
from torch_scatter import scatter_mean
from pointcept.utils.config import Config as PCConfig
from pointcept.datasets.transform import Compose
from pointcept.models import build_model

from data.datasets.scannet_base import ScanNetBase  
from data.datasets.scan_data_loader import ScanDataLoader
from data.datasets.msr3d import MSR3DBase
from data.datasets.msr3d import MSQAScanNet
from data.datasets.dataset_wrapper import LeoScanFamilyDatasetWrapper
from modules.vision.pcd_pointcept_encoder import PTv3PcdObjEncoder
from data.build import build_dataloader_leo
from model.msr3d.msr3d import MSR3D

# ✅ CHANGE THIS import to wherever your encoder code actually lives.
# Example:
# from modules.vision.ptv3_pcd_obj_encoder import transform_obj_pcds_to_pointcept
SCANNET20_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refridgerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
]

SCANNET20_TO_ID = {n: i for i, n in enumerate(SCANNET20_NAMES)}
def move_pointcept_data_to_device(data_dict, device):
    if isinstance(data_dict, torch.Tensor):
        return data_dict.to(device, non_blocking=True)
    if isinstance(data_dict, dict):
        return {k: move_pointcept_data_to_device(v, device) for k, v in data_dict.items()}
    if isinstance(data_dict, (list, tuple)):
        return type(data_dict)(move_pointcept_data_to_device(v, device) for v in data_dict)
    return data_dict

def _normalize_nyu40_class(name: str) -> str:
    """Normalize TSV nyu40class values to match config spelling."""
    name = (name or "").strip().lower()
    # TSV commonly uses "refrigerator" while Pointcept config uses misspelling "refridgerator"
    if name == "refrigerator":
        return "refridgerator"
    return name

def build_nyu40_to_scannet20_map(tsv_path: str) -> np.ndarray:
    """
    Returns an array `lut` such that lut[nyu40id] = scannet20id or -1 if ignored.
    Works for nyu40id in [0..max_nyu40id_in_tsv].
    """
    df = pd.read_csv(tsv_path, sep="\t")

    if "nyu40id" not in df.columns or "nyu40class" not in df.columns:
        raise ValueError(f"TSV missing required columns. Found: {df.columns.tolist()}")

    # Build dict nyu40id -> normalized name
    nyu40id_to_name = (
        df[["nyu40id", "nyu40class"]]
        .drop_duplicates()
        .dropna()
        .set_index("nyu40id")["nyu40class"]
        .to_dict()
    )

    max_id = int(max(nyu40id_to_name.keys()))
    lut = np.full((max_id + 1,), -1, dtype=np.int64)

    for nyu_id, name in nyu40id_to_name.items():
        key = _normalize_nyu40_class(name)
        if key in SCANNET20_TO_ID:
            lut[int(nyu_id)] = SCANNET20_TO_ID[key]

    return lut

def remap_nyu40_segment_to_scannet20(segment_nyu40: np.ndarray, lut: np.ndarray, ignore_index: int = -1) -> np.ndarray:
    """
    segment_nyu40: (N,) integer labels in NYU40 id space.
    lut: array from build_nyu40_to_scannet20_map(), where lut[nyu40id] -> scannet20id or -1.
    Returns: (N,) int64 labels in ScanNet20 id space (0..19) or ignore_index.
    """
    seg = np.asarray(segment_nyu40)
    if seg.ndim != 1:
        raise ValueError(f"segment must be 1D (N,), got shape {seg.shape}")

    out = np.full(seg.shape, ignore_index, dtype=np.int64)

    # only map valid ids within lut range
    valid = (seg >= 0) & (seg < lut.shape[0])
    out[valid] = lut[seg[valid].astype(np.int64)]

    # lut uses -1 for ignore; convert to ignore_index (usually -1 anyway)
    out[out < 0] = ignore_index
    return out

def prepare_test_data(idx, data_dict, transform, voxelize, post_transform, aug_transform):
        # load data
        data_dict = transform(data_dict)
        result_dict = dict(segment=data_dict.pop("segment"), name=data_dict.pop("name"))

        data_dict_list = []
        # for aug in aug_transform:
        #     data_dict_list.append(aug(deepcopy(data_dict)))
        print(f"Applying {len(aug_transform)} augmentations...")
        data_dict_list.append(aug_transform[0](deepcopy(data_dict)))

        fragment_list = []
        for data in data_dict_list:
            if voxelize is not None:
                data_part_list = voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            
            for data_part in data_part_list:
                data_part = [data_part]
                fragment_list += data_part

        for i in range(len(fragment_list)):
            fragment_list[i] = post_transform(fragment_list[i])

        result_dict["fragment_list"] = fragment_list
        return result_dict
def prepare_train_data(transform, data_dict):
        result_dict = transform(data_dict)
        return result_dict


def pool_features(obj_pcds):
    obj_features = []
    for obj in obj_pcds:
        obj_features.append(obj.mean(0))  # average over all points in an object
    return obj_features
def pool_features_scatter(obj_pcds):
    # obj_pcds: list of [Ni, D] tensors
    x = torch.cat(obj_pcds, dim=0)                       # [sum_i Ni, D]
    lengths = torch.tensor([t.size(0) for t in obj_pcds], device=x.device)
    obj_id = torch.repeat_interleave(
        torch.arange(len(obj_pcds), device=x.device),
        lengths
    )                                                    # [sum_i Ni]
    return scatter_mean(x, obj_id, dim=0)                # [num_obj, D]
def load_pointcept_checkpoint(model, weight_path, strict=False):
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    sd = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    model_keys = model.state_dict().keys()
    model_has_module = any(k.startswith("module.") for k in model_keys)
    ckpt_has_module = any(k.startswith("module.") for k in sd.keys())

    weight = OrderedDict()
    for k, v in sd.items():
        kk = k
        # Strip/add 'module.' to match the model
        if ckpt_has_module and not model_has_module and kk.startswith("module."):
            kk = kk[7:]
        elif (not ckpt_has_module) and model_has_module:
            kk = "module." + kk
        weight[kk] = v

    missing, unexpected = model.load_state_dict(weight, strict=strict)
    print(f"[PTv3 Checkpoint] Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    return model
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="msr3d/configs/data.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--scan_id", required=True)
    ap.add_argument("--num_points", type=int, default=1024, help="Points per object (subsample/repeat)")
    ap.add_argument("--grid_size", type=float, default=0.02)
    args = ap.parse_args()
    

    print("=== Original Scene ===")
    path = f"/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/{args.scan_id}.pth"
    #path = f"/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/{args.scan_id}.pth"
    data = torch.load(path, map_location="cpu",weights_only=False)  
    print("length:", len(data[0]))
    print(data[0].shape)

    cfg = OmegaConf.load(args.cfg)
    ds = ScanNetBase(cfg, split=args.split)
    dl = ScanDataLoader(cfg,'ScanNet')
    msr3d_base = MSR3DBase(cfg,'ScanNet')
    msqa_scannet = MSQAScanNet(cfg,'test')
    
    print(f"\n=== Load One Scan ===")
    # This path loads pcd_with_global_alignment + pcd_normals and builds obj_pcds as (Ni,9). 
    _, one_scan = ds._load_one_scan(args.scan_id, load_inst_info=True, load_pc_info=True)

    print(f"scan_id={args.scan_id} split={args.split}")
    print(one_scan.keys())
    print(f"Scene Pointcloud: {one_scan['scene_fts'].shape}")
    pcd = torch.load(f"/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/{args.scan_id}.pth", map_location="cpu",weights_only=False)
    #pcd = torch.load(f"/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/{args.scan_id}.pth", map_location="cpu",weights_only=False)
    print(f"Normal size: {len(pcd[0])}")
    # print(one_scan['inst_labels'])
    # print(len(one_scan['inst_labels']))
    # print(one_scan['inst_locs'])
    # print(one_scan["inst_locs"].shape)
    # print(one_scan['inst_colors'])
    # print(one_scan['inst_colors'][0].shape)
    # print(one_scan['obj_pcds'])
    # print(len(one_scan['obj_pcds']))
    # print(one_scan["obj_pcds"][1].shape)
    # print(len(one_scan['obj_center']))
    # print(len(one_scan['obj_center'][0]))
    
    # print(one_scan['obj_box_size'])
    # print(len(one_scan['obj_box_size']))
    # print(len(one_scan['obj_box_size'][0]))
    print(f"\n=== Get Data ===")
    data = dl.get_data('ScanNet',scan_id=args.scan_id,data_type=['obj_pcds'])
    print(data['obj_pcds'].keys())
    print(data['obj_pcds'][0].shape)
    print(f"\n=== Prepare data loading cache ===")
    cached_data = msr3d_base.prepare_data_loading_with_cache('ScanNet',scan_id=args.scan_id,data_type_list=['obj_pcds'])
    print(cached_data.keys())
    # print(cached_data['obj_pcds'].keys())
    # print(compare_scan_dict(data,cached_data))
    print(f"\n=== MSQA Scannet ===")
    scannet_data = msqa_scannet[1]
    print(scannet_data.keys())
    print(scannet_data['scene_fts'].shape)
    # print(f"Source: {scannet_data['source']}")
    print(f"scan_id: {scannet_data['scan_id']}")
    # print(f"obj_fts: {scannet_data['obj_fts'].shape}")
    # print(f"obj_locs: {scannet_data['obj_locs'].shape}")
    # print(f"img_fts: {scannet_data['img_fts'].shape}")
    # print(f"img_masks: {scannet_data['img_masks']}")
    # print(f"text_output: {scannet_data['text_output']}")
    # print(f"answer_list: {scannet_data['answer_list']}")
    # print(f"msr3d_prompt: {scannet_data['msr3d_prompt']}")
    # print(f"msr3d_imgs: {scannet_data['msr3d_imgs']}")
    # print(f"anchor_orientation: {scannet_data['anchor_orientation']}")
    # print(f"anchor_locs: {scannet_data['anchor_locs']}")
    # print(f"index: {scannet_data['index']}")
    # print(f"type: {scannet_data['type']}")
    # print(f"prompt_before_obj: {scannet_data['prompt_before_obj']}")
    # print(f"prompt_middle_1: {scannet_data['prompt_middle_1']}")
    # print(f"prompt_middle_2: {scannet_data['prompt_middle_2']}")
    # print(f"prompt_after_obj: {scannet_data['prompt_after_obj']}")
    print(f"\n=== After Dataset Wrap ===")
    wrapper = LeoScanFamilyDatasetWrapper(cfg, msqa_scannet, cfg.dataset_wrapper.args)
    wrapped_scannet_data = wrapper[1]
    print(wrapped_scannet_data.keys())
    print(wrapped_scannet_data['scene_fts'].shape)
    #print(wrapped_scannet_data['scene_mask'].shape)
    print(wrapped_scannet_data['obj_masks'].shape)
    #print(np.unique(wrapped_scannet_data['scene_mask']))
    print(np.unique(wrapped_scannet_data['obj_masks']))
    print(f"\n=== After Mask ===")
    #print(wrapped_scannet_data['scene_fts'][wrapped_scannet_data['scene_mask'] == 1].shape)

    
    #wrapped_scannet_data['scene_fts'] = wrapped_scannet_data['scene_fts'][wrapped_scannet_data['scene_mask'] == 1]
    print("=== After Grouping into objects ===")

    instance_labels = pcd[-1]
    # obj_pcds = []
    # for i in range(instance_labels.max() + 1):
    #     mask = instance_labels == i     # time consuming
    #     obj_pcds.append(wrapped_scannet_data['scene_fts'][mask])                    
    # wrapped_scannet_data['pooled_fts'] = obj_pcds
    # print(len(wrapped_scannet_data['pooled_fts']))
    
    # keep = (wrapped_scannet_data['obj_masks'] == 1)

    # wrapped_scannet_data['pooled_fts'] = [
    #     p for p, k in zip(wrapped_scannet_data['pooled_fts'], keep.tolist()) if k
    # ]
    # print(len(wrapped_scannet_data['pooled_fts']))

    # out = pool_features_scatter(wrapped_scannet_data['pooled_fts'])
    # print(out.shape)

    # PTv3 Pipeline
    ptv3_cfg = PCConfig.fromfile( "/home/panagiotis/msqa/Msqa_Thesis_2025/Pointcept_main/configs/scannet/semseg-pt-v3m1-1-ppt-extreme.py")
    #ptv3_cfg = PCConfig.fromfile( "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/Pointcept_main/configs/scannet/semseg-pt-v3m1-1-ppt-extreme.py")
    # transform_cfg = ptv3_cfg.data.test['transform']
    # voxelize_cfg = ptv3_cfg.data.test['test_cfg']['voxelize']
    # post_transform_cfg = ptv3_cfg.data.test['test_cfg']['post_transform']
    # aug_cfg = ptv3_cfg.data.test['test_cfg']['aug_transform']

    # transform = Compose(transform_cfg)
    # voxelize = Compose([voxelize_cfg])
    # post_transform = Compose(post_transform_cfg)
    # aug_transforms = [Compose(aug_transform) for aug_transform in aug_cfg]



    # transform_cfg = ptv3_cfg.data.train.datasets[1]['transform']
    # transform = Compose(transform_cfg)

    # coord = wrapped_scannet_data['scene_fts'][:,:3]
    # color = wrapped_scannet_data['scene_fts'][:,3:6]
    # normals = wrapped_scannet_data['scene_fts'][:,6:9]
    # condition = 'ScanNet'
    # data_dict ={
    #     'coord': coord,
    #     'color': color,
    #     'normal': normals,
    #     'condition': condition, 
    #     'name': args.scan_id,
    #     'inst_id' : instance_labels,
    # }
    # print(type(data_dict["coord"]), getattr(data_dict["coord"], "shape", None))
    # csv_path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/annotations/meta_data/scannetv2-labels.combined.tsv"
    # #csv_path = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/scannet_base/annotations/meta_data/scannetv2-labels.combined.tsv"
    # nyu40_to_scannet20_lut = build_nyu40_to_scannet20_map(csv_path)
    # scene_path = f"/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/{args.scan_id}.pth"
    # #scene_path = f"/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/{args.scan_id}.pth"
    # scene_data = torch.load(scene_path, map_location="cpu",weights_only=False)
    # data_dict["segment"] = scene_data[2]  # raw NYU40 segment
    # # after loading raw segment labels (NYU40 IDs)
    # data_dict["segment"] = remap_nyu40_segment_to_scannet20(data_dict["segment"], nyu40_to_scannet20_lut, ignore_index=-1)

    # # point_data = prepare_test_data(0, data_dict, transform, voxelize, post_transform, aug_transforms
    # point_data = prepare_train_data(transform, data_dict)
    # print("\n=== After Pointcept Transform ===")
    # print(f"Keys: {point_data.keys()}")
    # print(f"coord shape: {point_data['coord'].shape}")
    # print(f"grid coord shape: {point_data['grid_coord'].shape}")
    # print(f"feat shape: {point_data['feat'].shape}")
    # print(f"inst_id shape: {point_data['inst_id'].shape}")

    # model = build_model(ptv3_cfg.model)
    # weight_path = '/mnt/d/Thesis/PTv3/model_best.pth'
    # #weight_path = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/modules/third_party/PTv3/model_best.pth"
    # model = load_pointcept_checkpoint(model, weight_path, strict=False)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # point_data = move_pointcept_data_to_device(point_data, device)
    # core = model.eval().to(device)
    # with torch.no_grad():
    #     out = core.backbone(point_data)
    # print(out.keys())

    # obj_pcds = []
    # unique_inst_ids = torch.unique(out['inst_id'])
    # for i in unique_inst_ids:
    #     mask = out['inst_id'] == i     # time consuming
    #     obj_pcds.append(out['feat'][mask])                    
    # out['pooled_fts'] = obj_pcds
    # print(len(out['pooled_fts']))
    

    # keep = (wrapped_scannet_data['obj_masks'] == 1)

    # out['pooled_fts'] = [
    #     p for p, k in zip(out['pooled_fts'], keep.tolist()) if k
    # ]
    # print(len(out['pooled_fts']))

    # out = pool_features_scatter(out['pooled_fts'])
    # print(out.shape)
    #model = MSR3D(cfg)
    task_name = "msr3d_train"
    mode = "train"
    loader = build_dataloader_leo(cfg,
                                    cfg.task[task_name].dataset,
                                    cfg.task[task_name].dataset_wrapper,
                                    cfg.task[task_name].dataset_wrapper_args,
                                    cfg.task[task_name].train_dataloader_args if mode == "train" else cfg.task[task_name].eval_dataloader_args,
                                    split=mode,)
    
    batch = next(iter(loader))
    print(batch.keys())
    ptv3 = PTv3PcdObjEncoder(cfg = cfg,
                            embedding_size = cfg.model.prompter.model.vision.args.embedding_size,
                            sem_num_classes = cfg.model.prompter.model.vision.args.sem_num_classes,
                            ptv3_cfg_path = cfg.model.prompter.model.vision.args.ptv3_cfg_path, 
                            weight_path = cfg.model.prompter.model.vision.args.weight_path,
                            grid_size = cfg.model.prompter.model.vision.args.grid_size,
                            feat_reduce = cfg.model.prompter.model.vision.args.feat_reduce,
                            freeze=True)


    print(f"Scene Features Shape: {batch['scene_fts'].shape}")
    #print(f"Object instance Shape: {batch['instance_ids'].shape}")
    print(f"Object segment Shape: {batch['segments'].shape}")
    
    obj_embeds,obj_logits = ptv3(batch, mode = "inference")
    print(f"Object Embeddings Shape: {obj_embeds.shape}")
    print(f"Object Logits Shape: {obj_logits.shape}")

    # print(f"Shape of point_data['fragment_list']: {len(point_data['fragment_list'])}")
    # for i, frag in enumerate(point_data['fragment_list']):
    #     print(f"\n--- Fragment {i} ---")
    #     print(f"Keys: {frag.keys()}")
    #     for k, v in frag.items():
    #         if torch.is_tensor(v):
    #             print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
    #         else:
    #             print(f"{k}: {v}")
    # print(point_data['segment'])
    # print(point_data['name'])
    
if __name__ == "__main__":
    main()
