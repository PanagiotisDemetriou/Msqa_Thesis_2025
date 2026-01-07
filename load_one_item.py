# #!/usr/bin/env python3
# """
# Quick sanity test for the EXACT dataset + wrapper path your MSR3D config uses.

# It:
# - loads your YAML config
# - builds the dataset specified in cfg.task.msr3d_train (MSR3DMix)
# - wraps it with LeoScanFamilyDatasetWrapper (as in your config)
# - calls __getitem__ for a few indices
# - prints shapes/dtypes and validates normals if present (expects normals at cols 6:9)
# """

# import os
# import sys
# import argparse
# import random
# import numpy as np
# import torch

# # -----------------------------
# # Adjust these imports to your repo layout
# # -----------------------------
# # Typically you have something like:
# #   from datasets.default import DATASET_REGISTRY
# #   from datasets.dataset_wrapper import DATASETWRAPPER_REGISTRY
# #
# # In your snippets they are:
# #   from .default import DATASET_REGISTRY
# #   DATASETWRAPPER_REGISTRY = Registry("dataset_wrapper")
# #
# # So for a script at repo root, it is often:
# #   from <your_pkg>.datasets.default import DATASET_REGISTRY
# #   from <your_pkg>.datasets.dataset_wrapper import DATASETWRAPPER_REGISTRY
# #
# # Edit the two lines below to match your project.
# from msr3d.data.datasets.default import DATASET_REGISTRY
# from msr3d.data.datasets.dataset_wrapper import DATASETWRAPPER_REGISTRY

# # OmegaConf is typically used with these configs
# from omegaconf import OmegaConf


# def build_from_registry(registry, name, *args, **kwargs):
#     if name not in registry._obj_map:
#         raise KeyError(f"'{name}' not found in registry. Available: {list(registry._obj_map.keys())}")
#     cls = registry.get(name)
#     return cls(*args, **kwargs)


# def pretty(x):
#     if isinstance(x, torch.Tensor):
#         return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})"
#     if isinstance(x, (list, tuple)):
#         return f"{type(x).__name__}(len={len(x)})"
#     return f"{type(x).__name__}"


# @torch.no_grad()
# def inspect_item(item, idx):
#     print("=" * 80)
#     print(f"Index: {idx}")
#     for k, v in item.items():
#         if isinstance(v, torch.Tensor):
#             s = f"{k:20s}: {pretty(v)}"
#             if v.numel() > 0 and v.dtype.is_floating_point:
#                 s += f", min={v.min().item():.4g}, max={v.max().item():.4g}"
#             print(s)
#         elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
#             print(f"{k:20s}: list[tensor] len={len(v)}, first={pretty(v[0])}")
#         else:
#             # keep strings short
#             if isinstance(v, str) and len(v) > 160:
#                 v_show = v[:160] + "..."
#             else:
#                 v_show = v
#             print(f"{k:20s}: {pretty(v)} -> {v_show}")

#     # Specific checks for obj_fts
#     if "obj_fts" in item and isinstance(item["obj_fts"], torch.Tensor):
#         obj_fts = item["obj_fts"]
#         if obj_fts.ndim != 3:
#             print("[WARN] obj_fts is not 3D (O, P, C).")
#         else:
#             O, P, C = obj_fts.shape
#             print(f"\nobj_fts channels: C={C} (expected 6 for xyzrgb or 9 for xyzrgb+normals)")
#             if C >= 9:
#                 n = obj_fts[..., 6:9]
#                 n_norm = torch.linalg.norm(n, dim=-1)
#                 # mask padding rows if present
#                 if "obj_masks" in item and isinstance(item["obj_masks"], torch.Tensor) and item["obj_masks"].ndim == 1:
#                     valid = item["obj_masks"].bool()
#                     n_norm = n_norm[valid]
#                 # avoid empty
#                 if n_norm.numel() > 0:
#                     print(f"Normals ||n||: mean={n_norm.mean().item():.4f}, "
#                           f"min={n_norm.min().item():.4f}, max={n_norm.max().item():.4f}")
#             else:
#                 print("[INFO] obj_fts has no normals (C < 9).")

#     # Check for your custom key if you kept it
#     if "obj_normals" in item:
#         print("\n[INFO] Found obj_normals key:", pretty(item["obj_normals"]))


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--cfg", required=True, help="Path to your YAML config")
#     ap.add_argument("--split", default="train", choices=["train", "val", "test"])
#     ap.add_argument("--num", type=int, default=3, help="How many samples to test")
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--indices", nargs="*", type=int, default=None, help="Explicit indices to test")
#     args = ap.parse_args()

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     cfg = OmegaConf.load(args.cfg)

#     # The exact dataset+wrapper used by MSR3D in your config:
#     task_cfg = cfg.task.msr3d_train
#     dataset_name = task_cfg.dataset                      # MSR3DMix
#     wrapper_name = task_cfg.dataset_wrapper              # LeoScanFamilyDatasetWrapper
#     wrapper_args = task_cfg.get("dataset_wrapper_args", {})  # cfg.dataset_wrapper.args

#     print("Dataset (from cfg.task.msr3d_train):", dataset_name)
#     print("Wrapper (from cfg.task.msr3d_train):", wrapper_name)

#     # Build dataset
#     dataset = build_from_registry(DATASET_REGISTRY, dataset_name, cfg, args.split)

#     # Wrap dataset
#     wrapper = build_from_registry(DATASETWRAPPER_REGISTRY, wrapper_name, cfg, dataset, wrapper_args)

#     print(f"Wrapped dataset length: {len(wrapper)}")

#     # Pick indices
#     if args.indices is not None and len(args.indices) > 0:
#         indices = args.indices
#     else:
#         indices = [0, min(1, len(wrapper) - 1), min(2, len(wrapper) - 1)]
#         indices = indices[: max(1, min(args.num, len(indices)))]
#         # If you asked for more than 3, extend deterministically
#         while len(indices) < min(args.num, len(wrapper)):
#             indices.append(len(indices))

#     # Run __getitem__
#     for idx in indices:
#         item = wrapper[idx]
#         if not isinstance(item, dict):
#             raise TypeError(f"Expected dict from __getitem__, got {type(item)}")
#         inspect_item(item, idx)

#     print("\nDone.")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Sanity test:
- loads YAML config
- builds MSR3DMix + LeoScanFamilyDatasetWrapper (as your config specifies)
- calls __getitem__ for a few indices
- runs PTv3PcdObjEncoder on item["obj_fts"] (expects xyzrgb+normals => C>=9)
- prints embedding / logits shapes and basic normal statistics
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch_scatter
from torch import nn
from collections import OrderedDict
from omegaconf import OmegaConf

# Project registries (adjust if your repo layout differs)
from data.datasets.default import DATASET_REGISTRY
from data.datasets.dataset_wrapper import DATASETWRAPPER_REGISTRY

# Pointcept imports (must be available in PYTHONPATH)
from pointcept.utils.config import Config as PCConfig
from pointcept.models import build_model
from pointcept.models.utils import batch2offset
import pointcept.utils.comm as comm

# Optional: if you want sem head like your other encoder
from modules.utils import get_mlp_head
import modules.vision as vision

# -----------------------------
# Registry helper
# -----------------------------
def build_from_registry(registry, name, *args, **kwargs):
    if name not in registry._obj_map:
        raise KeyError(f"'{name}' not found in registry. Available: {list(registry._obj_map.keys())}")
    cls = registry.get(name)
    return cls(*args, **kwargs)


def pretty(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}(len={len(x)})"
    return f"{type(x).__name__}"


# -----------------------------
# Pointcept utilities
# -----------------------------
def move_pointcept_data_to_device(data_dict, device):
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
        if not k.startswith("module."):
            k = "module." + k
        if comm.get_world_size() == 1:
            k = k[7:]
        weight[k] = v

    missing, unexpected = model.load_state_dict(weight, strict=strict)
    print(f"[PTv3 Checkpoint] Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    return model


def transform_obj_pcds_to_pointcept(obj_pcds, grid_size=0.02, rgb_div=255.0):
    """
    obj_pcds: (B, O, P, >=9) with [xyz(3), rgb(3), normals(3), ...]
    returns point_data dict for PTv3 backbone.
    """
    if obj_pcds.ndim != 4:
        raise ValueError(f"obj_pcds must be (B,O,P,C). Got {tuple(obj_pcds.shape)}")

    B, O, P, C = obj_pcds.shape
    if C < 9:
        raise ValueError(f"Expected >=9 channels [xyz,rgb,normals]. Got C={C}")

    total_points = B * O * P

    coord = obj_pcds[..., :3].reshape(total_points, 3)
    rgb = obj_pcds[..., 3:6].reshape(total_points, 3) / float(rgb_div)
    normals = obj_pcds[..., 6:9].reshape(total_points, 3)

    # Pointcept feature = [rgb, normals] => (N_total, 6)
    feat = torch.cat([rgb, normals], dim=1)

    batch = (
        torch.arange(B, device=obj_pcds.device)
        .view(B, 1, 1)
        .expand(B, O, P)
        .reshape(total_points)
        .long()
    )

    obj_id = (
        torch.arange(B * O, device=obj_pcds.device)
        .repeat_interleave(P)
        .long()
    )

    point_data = {
        "coord": coord,
        "feat": feat,
        "batch": batch,
        "offset": batch2offset(batch),
        "grid_size": float(grid_size),
        "obj_id": obj_id,
        "condition": ["ScanNet"],
    }
    return point_data


def pool_point_features_to_objects(point_feats, obj_id, num_objs, reduce="mean"):
    if point_feats.ndim != 2:
        raise ValueError(f"point_feats must be (N,F). Got {tuple(point_feats.shape)}")
    if obj_id.ndim != 1:
        raise ValueError(f"obj_id must be (N,). Got {tuple(obj_id.shape)}")
    if point_feats.shape[0] != obj_id.shape[0]:
        raise ValueError("point_feats and obj_id must have same first dimension")
    if reduce not in {"mean", "max", "sum", "min"}:
        raise ValueError("reduce must be one of mean/max/sum/min")

    return torch_scatter.scatter(
        src=point_feats,
        index=obj_id.long(),
        dim=0,
        dim_size=int(num_objs),
        reduce=reduce,
    )


# -----------------------------
# PTv3 object encoder (test version: no proj head)
# -----------------------------
class PTv3PcdObjEncoder(nn.Module):
    """
    Object-level encoder using PointTransformerV3 backbone (Pointcept).
    Produces per-object embeddings by pooling per-point backbone features.
    """
    def __init__(
        self,
        cfg,
        ptv3_cfg_path: str,
        weight_path: str = None,
        grid_size: float = 0.02,
        feat_reduce: str = "mean",
        out_dim: int = None,          
        sem_num_classes: int = None,  
        dropout: float = 0.1,
        freeze: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.grid_size = float(grid_size)
        self.feat_reduce = feat_reduce
        self.freeze = freeze

        # Build Pointcept model from config file
        self.ptv3_cfg = PCConfig.fromfile(ptv3_cfg_path)
        model = build_model(self.ptv3_cfg.model)
       
        if weight_path is not None:
            model = load_pointcept_checkpoint(model, weight_path, strict=False)

        # Keep only the backbone path you already validated
        self.model = model
        self.dropout = nn.Dropout(dropout)

        # Optional projection head to match a target dimension
        self.proj = None
        # self.proj_out_dim = None
        # if out_dim is not None:
        #     self.proj_out_dim = int(out_dim)
        #     # We can’t know backbone dim until first forward unless you hardcode.
        #     # So we create proj lazily.
        #     self._lazy_proj = True
        # else:
        #     self._lazy_proj = False

        # Optional semantic classifier head (like your obj3d_clf_pre_head)
        self.sem_num_classes = 607
        self.sem_head = get_mlp_head(64, 384, self.sem_num_classes, dropout=0.3)
        if sem_num_classes is not None:
            self.sem_num_classes = int(sem_num_classes)
            self._lazy_sem = True
        else:
            self._lazy_sem = False

        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _get_core(self):
        core = self.model.module if hasattr(self.model, "module") else self.model
        return core

    # def _maybe_build_heads(self, in_dim: int):
    #     if self._lazy_proj and self.proj is None:
    #         self.proj = nn.Sequential(
    #             nn.Linear(in_dim, self.proj_out_dim),
    #             nn.ReLU(inplace=True),
    #         )
    #         self._lazy_proj = False

    #     if self._lazy_sem and self.sem_head is None:
    #         # mirror your style: MLP head to sem classes
    #         # (replace 384 with whatever your downstream expects if needed)
    #         self.sem_head = get_mlp_head(in_dim, 384, self.sem_num_classes, dropout=0.3)
    #         self._lazy_sem = False

    def forward(self, obj_pcds, obj_locs=None, obj_masks=None, obj_sem_masks=None, **kwargs):
        """
        obj_pcds: (B,O,P,9) [xyz,rgb,normals]
        """
        B, O, P, C = obj_pcds.shape
        device = obj_pcds.device

        point_data = transform_obj_pcds_to_pointcept(
            obj_pcds=obj_pcds,
            grid_size=self.grid_size,
        )
        point_data = move_pointcept_data_to_device(point_data, device)

        core = self._get_core().to(device).eval() if self.freeze else self._get_core().to(device)

        if self.freeze:
            with torch.no_grad():
                point_out = core.backbone(point_data)
        else:
            point_out = core.backbone(point_data)

        obj_feats = pool_point_features_to_objects(
            point_feats=point_out.feat,
            obj_id=point_data["obj_id"],
            num_objs=B * O,
            reduce=self.feat_reduce,
        )

        # D = int(obj_feats.shape[-1])
        # self._maybe_build_heads(D)

        obj_feats = self.dropout(obj_feats)
        if self.proj is not None:
            obj_feats = self.proj(obj_feats)

        obj_embeds = obj_feats.view(B, O, -1)

        obj_sem_cls = None
        if self.sem_head is not None:
            obj_sem_cls = self.sem_head(obj_embeds)
        print("pqpqpqpqpqpqpqpqpqpqpqpqpqpqpqpqpqpqpqpqpqp")
        return obj_embeds, obj_sem_cls




# -----------------------------
# Dataset inspection + encoder run
# -----------------------------
@torch.no_grad()
def inspect_item(item, idx):
    print("=" * 80)
    print(f"Index: {idx}")
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            s = f"{k:20s}: {pretty(v)}"
            if v.numel() > 0 and v.dtype.is_floating_point:
                s += f", min={v.min().item():.4g}, max={v.max().item():.4g}"
            print(s)
        else:
            v_show = (v[:160] + "...") if isinstance(v, str) and len(v) > 160 else v
            print(f"{k:20s}: {pretty(v)} -> {v_show}")

    if "obj_fts" in item and isinstance(item["obj_fts"], torch.Tensor):
        obj_fts = item["obj_fts"]
        if obj_fts.ndim == 3:
            O, P, C = obj_fts.shape
            print(f"\nobj_fts: (O={O}, P={P}, C={C})")
            if C >= 9:
                n = obj_fts[..., 6:9]
                n_norm = torch.linalg.norm(n, dim=-1)
                if "obj_masks" in item and isinstance(item["obj_masks"], torch.Tensor):
                    # support (O,) or (B,O)
                    m = item["obj_masks"]
                    if m.ndim == 1 and m.shape[0] == O:
                        n_norm = n_norm[m.bool()]
                if n_norm.numel() > 0:
                    print(f"Normals ||n||: mean={n_norm.mean().item():.4f}, "
                          f"min={n_norm.min().item():.4f}, max={n_norm.max().item():.4f}")
            else:
                print("[WARN] obj_fts has C<9; encoder will fail (no normals).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to your YAML config")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--num", type=int, default=3, help="How many samples to test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--indices", nargs="*", type=int, default=None)

    # Encoder args
    ap.add_argument("--ptv3_cfg", required=True, help="Pointcept PTv3 config .py path")
    ap.add_argument("--ptv3_ckpt", default=None, help="Pointcept checkpoint .pth")
    ap.add_argument("--sem_classes", type=int, default=None, help="Optional sem head classes")
    ap.add_argument("--freeze", action="store_true", help="Freeze PTv3 backbone for test")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.cfg)

    task_cfg = cfg.task.msr3d_train
    dataset_name = task_cfg.dataset
    wrapper_name = task_cfg.dataset_wrapper
    wrapper_args = task_cfg.get("dataset_wrapper_args", {})

    print("Dataset (cfg.task.msr3d_train):", dataset_name)
    print("Wrapper (cfg.task.msr3d_train):", wrapper_name)

    dataset = build_from_registry(DATASET_REGISTRY, dataset_name, cfg, args.split)
    wrapper = build_from_registry(DATASETWRAPPER_REGISTRY, wrapper_name, cfg, dataset, wrapper_args)

    print(f"Wrapped dataset length: {len(wrapper)}")

    # Select indices
    if args.indices:
        indices = args.indices
    else:
        indices = [0, min(1, len(wrapper) - 1), min(2, len(wrapper) - 1)]
        indices = indices[: max(1, min(args.num, len(indices)))]

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    # Build encoder once
    encoder = PTv3PcdObjEncoder(
        cfg=cfg,
        ptv3_cfg_path=args.ptv3_cfg,
        weight_path=args.ptv3_ckpt,
        sem_num_classes=args.sem_classes,
        freeze=args.freeze,
    ).to(device)
    encoder.eval()

    # Run samples
    for idx in indices:
        item = wrapper[idx]
        if not isinstance(item, dict):
            raise TypeError(f"Expected dict from __getitem__, got {type(item)}")

        inspect_item(item, idx)

        if "obj_fts" not in item:
            print("[SKIP] No obj_fts in item; cannot run encoder.")
            continue

        obj_fts = item["obj_fts"]
        if not isinstance(obj_fts, torch.Tensor) or obj_fts.ndim != 3:
            print("[SKIP] obj_fts not a 3D tensor (O,P,C).")
            continue

        # Add batch dimension: (1,O,P,C)
        obj_pcds = obj_fts.unsqueeze(0).to(device)

        # Optional mask
        obj_masks = item.get("obj_masks", None)
        if isinstance(obj_masks, torch.Tensor) and obj_masks.ndim == 1:
            obj_masks_b = obj_masks.unsqueeze(0).to(device)
        else:
            obj_masks_b = None

        with torch.no_grad():
            obj_embeds, obj_sem = encoder(obj_pcds, obj_masks=obj_masks_b)

        print("\n[ENCODER OUTPUT]")
        print("obj_embeds:", pretty(obj_embeds))
        if obj_sem is not None:
            print("obj_sem_cls:", pretty(obj_sem))
        else:
            print("obj_sem_cls: None")

    print("\nDone.")


if __name__ == "__main__":
    main()
