import einops
from omegaconf import OmegaConf
import torch
from torch import nn
import torch_scatter
from collections import OrderedDict
from triton import Config

from modules.build import VISION_REGISTRY
from modules.utils import get_mlp_head

from pointcept.utils.config import Config as PCConfig
from pointcept.models import build_model
from pointcept.models.utils import batch2offset
import pointcept.utils.comm as comm

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

# def transform_obj_pcds_to_pointcept(obj_pcds, grid_size=0.02):
#     B, O, P, C = obj_pcds.shape
#     if C < 9:
#         raise ValueError(f"Expected >=9 channels [xyz,rgb,normals]. Got {C}")

#     total = B * O * P
#     coord = obj_pcds[..., :3].reshape(total, 3)

#     rgb = obj_pcds[..., 3:6].reshape(total, 3)
#     rgb_min = float(rgb.min())
#     rgb_max = float(rgb.max())
#     if rgb_min < 0.0:
#         rgb = (rgb + 1.0) * 0.5
#     elif rgb_max > 1.5:
#         rgb = rgb / 255.0
#     rgb = rgb.clamp(0.0, 1.0)

#     normals = obj_pcds[..., 6:9].reshape(total, 3)
#     normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-6)

#     feat = torch.cat([rgb, normals], dim=1)

#     obj_id = torch.arange(B * O, device=obj_pcds.device).repeat_interleave(P).long()

#     # CRITICAL: treat each object as its own sample
#     batch = obj_id
#     offset = batch2offset(batch)

#     return {
#         "coord": coord,
#         "feat": feat,
#         "batch": batch,
#         "offset": offset,
#         "grid_size": float(grid_size),
#         "obj_id": obj_id,
#         "condition": ["ScanNet"],
#     }
def transform_obj_pcds_to_pointcept(scene_pcd, grid_size=0.02):
    """
    mode:
      - "object": each object is a separate Pointcept sample (current behavior)
      - "scene":  all objects within each scene are merged; PTv3 runs once per scene
    """
    if scene_pcd.dim() != 3:
        raise ValueError(f"Expected scene_pcd (B,P,C). Got {tuple(scene_pcd.shape)}")

    B, P, C = scene_pcd.shape
    if C < 9:
        raise ValueError(f"Expected >=9 channels [xyz,rgb,normals]. Got {C}")

    coord = scene_pcd[..., :3]
    rgb = scene_pcd[..., 3:6]
    normals = scene_pcd[..., 6:9]
    feat = torch.cat([rgb, normals], dim=1)  # (N, 6)

    
    batch = torch.arange(B, device=scene_pcd.device).repeat_interleave(P).long()
    offset = batch2offset(batch)

    return {
        "coord": coord,
        "feat": feat,
        "batch": batch,
        "offset": offset,
        "grid_size": float(grid_size),
        "obj_id": obj_id,
        "condition": ["ScanNet"],
    }

def pool_point_features_to_objects(point_feats, obj_id, num_objs, reduce="mean"):
    """
    point_feats: (N_total, F)
    obj_id: (N_total,) long in [0..num_objs-1]
    """
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


@VISION_REGISTRY.register()
class PTv3PcdObjEncoder(nn.Module):
    """
    Object-level encoder using PointTransformerV3 backbone (Pointcept).
    Produces per-object embeddings by pooling per-point backbone features.
    """
    def __init__(
        self,
        cfg,
        embedding_size: int,
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

        # Optional semantic classifier head (like your obj3d_clf_pre_head)
        self.sem_num_classes = sem_num_classes
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

    
    def forward(self, scene_pcd, obj_locs=None, obj_masks=None, obj_sem_masks=None, **kwargs):
        """
        obj_pcds: (B,O,P,9) [xyz,rgb,normals]
        obj_masks: (B,O) bool, True=valid object, False=padding
        """
        B, P, C = scene_pcd.shape
        device = scene_pcd.device


        # Transform to scene-level batching
        point_data = transform_obj_pcds_to_pointcept(
            scene_pcd=scene_pcd,  # Use cleaned version
            grid_size=self.grid_size,  
        )
        point_data = move_pointcept_data_to_device(point_data, device)

        # Process through PTv3 backbone
        core = self._get_core().to(device).eval() if self.freeze else self._get_core().to(device)

        if self.freeze:
            with torch.no_grad():
                point_out = core.backbone(point_data)
        else:
            point_out = core.backbone(point_data)

        # Pool point features to object features
        obj_feats = pool_point_features_to_objects(
            point_feats=point_out.feat,
            obj_id=point_data["obj_id"],
            num_objs=B * O,
            reduce=self.feat_reduce,
        )  # (B*O, F_backbone)

        # Reshape to (B, O, F)
        obj_embeds = obj_feats.view(B, O, -1)
        
        # CRITICAL: Mask out padded objects before any further processing
        obj_embeds = obj_embeds * obj_masks.unsqueeze(-1).to(obj_embeds.dtype)
        
        # Apply dropout
        #obj_embeds = self.dropout(obj_embeds)

        # Semantic classification head
        obj_sem_cls = None
        if self.sem_head is not None:
            obj_sem_cls = self.sem_head(obj_embeds)
            # Optional: also mask semantic logits if your loss doesn't handle it
            # obj_sem_cls = obj_sem_cls * obj_masks.unsqueeze(-1).to(obj_sem_cls.dtype)
        
        return obj_embeds, obj_sem_cls
# import einops
# from omegaconf import OmegaConf
# import torch
# from torch import nn
# import torch_scatter
# from collections import OrderedDict
# from triton import Config

# from modules.build import VISION_REGISTRY
# from modules.utils import get_mlp_head

# from pointcept.utils.config import Config as PCConfig
# from pointcept.models import build_model
# from pointcept.models.utils import batch2offset
# import pointcept.utils.comm as comm

# def move_pointcept_data_to_device(data_dict, device):
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

#     model_keys = model.state_dict().keys()
#     model_has_module = any(k.startswith("module.") for k in model_keys)
#     ckpt_has_module = any(k.startswith("module.") for k in sd.keys())

#     weight = OrderedDict()
#     for k, v in sd.items():
#         kk = k
#         # Strip/add 'module.' to match the model
#         if ckpt_has_module and not model_has_module and kk.startswith("module."):
#             kk = kk[7:]
#         elif (not ckpt_has_module) and model_has_module:
#             kk = "module." + kk
#         weight[kk] = v

#     missing, unexpected = model.load_state_dict(weight, strict=strict)
#     print(f"[PTv3 Checkpoint] Missing: {len(missing)}  Unexpected: {len(unexpected)}")
#     return model

# # def transform_obj_pcds_to_pointcept(obj_pcds, grid_size=0.02):
# #     B, O, P, C = obj_pcds.shape
# #     if C < 9:
# #         raise ValueError(f"Expected >=9 channels [xyz,rgb,normals]. Got {C}")

# #     total = B * O * P
# #     coord = obj_pcds[..., :3].reshape(total, 3)

# #     rgb = obj_pcds[..., 3:6].reshape(total, 3)
# #     rgb_min = float(rgb.min())
# #     rgb_max = float(rgb.max())
# #     if rgb_min < 0.0:
# #         rgb = (rgb + 1.0) * 0.5
# #     elif rgb_max > 1.5:
# #         rgb = rgb / 255.0
# #     rgb = rgb.clamp(0.0, 1.0)

# #     normals = obj_pcds[..., 6:9].reshape(total, 3)
# #     normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-6)

# #     feat = torch.cat([rgb, normals], dim=1)

# #     obj_id = torch.arange(B * O, device=obj_pcds.device).repeat_interleave(P).long()

# #     # CRITICAL: treat each object as its own sample
# #     batch = obj_id
# #     offset = batch2offset(batch)

# #     return {
# #         "coord": coord,
# #         "feat": feat,
# #         "batch": batch,
# #         "offset": offset,
# #         "grid_size": float(grid_size),
# #         "obj_id": obj_id,
# #         "condition": ["ScanNet"],
# #     }
# def transform_obj_pcds_to_pointcept(obj_pcds, grid_size=0.02, mode="scene"):
#     """
#     mode:
#       - "object": each object is a separate Pointcept sample (current behavior)
#       - "scene":  all objects within each scene are merged; PTv3 runs once per scene
#     """
#     if obj_pcds.dim() != 4:
#         raise ValueError(f"Expected obj_pcds (B,O,P,C). Got {tuple(obj_pcds.shape)}")

#     B, O, P, C = obj_pcds.shape
#     if C < 9:
#         raise ValueError(f"Expected >=9 channels [xyz,rgb,normals]. Got {C}")

#     total = B * O * P
#     coord = obj_pcds[..., :3].reshape(total, 3)

#     rgb = obj_pcds[..., 3:6].reshape(total, 3)
#     rgb_min = float(rgb.min())
#     rgb_max = float(rgb.max())
#     if rgb_min < 0.0:
#         rgb = (rgb + 1.0) * 0.5
#     elif rgb_max > 1.5:
#         rgb = rgb / 255.0
#     rgb = rgb.clamp(0.0, 1.0)

#     normals = obj_pcds[..., 6:9].reshape(total, 3)
#     normals = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-6)

#     feat = torch.cat([rgb, normals], dim=1)  # (N, 6)

#     # Object id for pooling back to objects:
#     # points are ordered as: scene0 obj0, scene0 obj1, ..., scene1 obj0, ...
#     obj_id = torch.arange(B * O, device=obj_pcds.device).repeat_interleave(P).long()

#     if mode == "object":
#         batch = obj_id  # B*O samples
#     elif mode == "scene":
#         # scene id: [0..B-1], each repeated (O*P) times
#         batch = torch.arange(B, device=obj_pcds.device).repeat_interleave(O * P).long()
#     else:
#         raise ValueError(f"mode must be 'object' or 'scene'. Got {mode}")

#     offset = batch2offset(batch)

#     return {
#         "coord": coord,
#         "feat": feat,
#         "batch": batch,
#         "offset": offset,
#         "grid_size": float(grid_size),
#         "obj_id": obj_id,
#         "condition": ["ScanNet"],
#     }

# def pool_point_features_to_objects(point_feats, obj_id, num_objs, reduce="mean"):
#     """
#     point_feats: (N_total, F)
#     obj_id: (N_total,) long in [0..num_objs-1]
#     """
#     if point_feats.ndim != 2:
#         raise ValueError(f"point_feats must be (N,F). Got {tuple(point_feats.shape)}")
#     if obj_id.ndim != 1:
#         raise ValueError(f"obj_id must be (N,). Got {tuple(obj_id.shape)}")
#     if point_feats.shape[0] != obj_id.shape[0]:
#         raise ValueError("point_feats and obj_id must have same first dimension")
#     if reduce not in {"mean", "max", "sum", "min"}:
#         raise ValueError("reduce must be one of mean/max/sum/min")

#     return torch_scatter.scatter(
#         src=point_feats,
#         index=obj_id.long(),
#         dim=0,
#         dim_size=int(num_objs),
#         reduce=reduce,
#     )


# @VISION_REGISTRY.register()
# class PTv3PcdObjEncoder(nn.Module):
#     """
#     Object-level encoder using PointTransformerV3 backbone (Pointcept).
#     Produces per-object embeddings by pooling per-point backbone features.
#     """
#     def __init__(
#         self,
#         cfg,
#         embedding_size: int,
#         ptv3_cfg_path: str,
#         weight_path: str = None,
#         grid_size: float = 0.02,
#         feat_reduce: str = "mean",
#         out_dim: int = None,          
#         sem_num_classes: int = None,  
#         dropout: float = 0.1,
#         freeze: bool = False,
#     ):
#         super().__init__()
#         self.cfg = cfg
#         self.grid_size = float(grid_size)
#         self.feat_reduce = feat_reduce
#         self.freeze = freeze

#         # Build Pointcept model from config file
#         self.ptv3_cfg = PCConfig.fromfile(ptv3_cfg_path)
#         model = build_model(self.ptv3_cfg.model)
       
#         if weight_path is not None:
#             model = load_pointcept_checkpoint(model, weight_path, strict=False)

#         # Keep only the backbone path you already validated
#         self.model = model
#         self.dropout = nn.Dropout(dropout)

#         # Optional semantic classifier head (like your obj3d_clf_pre_head)
#         self.sem_num_classes = sem_num_classes
#         self.sem_head = get_mlp_head(64, 384, self.sem_num_classes, dropout=0.3)
#         if sem_num_classes is not None:
#             self.sem_num_classes = int(sem_num_classes)
#             self._lazy_sem = True
#         else:
#             self._lazy_sem = False

#         if self.freeze:
#             for p in self.parameters():
#                 p.requires_grad = False

#     def _get_core(self):
#         core = self.model.module if hasattr(self.model, "module") else self.model
#         return core

    
#     def forward(self, obj_pcds, obj_locs=None, obj_masks=None, obj_sem_masks=None, **kwargs):
#         """
#         obj_pcds: (B,O,P,9) [xyz,rgb,normals]
#         obj_masks: (B,O) bool, True=valid object, False=padding
#         """
#         B, O, P, C = obj_pcds.shape
#         device = obj_pcds.device

#         # Create mask if not provided
#         if obj_masks is None:
#             obj_masks = torch.ones((B, O), device=device, dtype=torch.bool)
#         else:
#             obj_masks = obj_masks.to(device=device, dtype=torch.bool)

#         # Optional: Zero out padded object point clouds to avoid garbage data
#         # This ensures PTv3 processes clean data even for padding
#         obj_pcds_clean = obj_pcds.clone()
#         obj_pcds_clean = obj_pcds_clean * obj_masks.view(B, O, 1, 1)

#         # Transform to scene-level batching
#         point_data = transform_obj_pcds_to_pointcept(
#             obj_pcds=obj_pcds_clean,  # Use cleaned version
#             grid_size=self.grid_size,
#             mode="scene",   
#         )
#         point_data = move_pointcept_data_to_device(point_data, device)

#         # Process through PTv3 backbone
#         core = self._get_core().to(device).eval() if self.freeze else self._get_core().to(device)

#         if self.freeze:
#             with torch.no_grad():
#                 point_out = core.backbone(point_data)
#         else:
#             point_out = core.backbone(point_data)

#         # Pool point features to object features
#         obj_feats = pool_point_features_to_objects(
#             point_feats=point_out.feat,
#             obj_id=point_data["obj_id"],
#             num_objs=B * O,
#             reduce=self.feat_reduce,
#         )  # (B*O, F_backbone)

#         # Reshape to (B, O, F)
#         obj_embeds = obj_feats.view(B, O, -1)
        
#         # CRITICAL: Mask out padded objects before any further processing
#         obj_embeds = obj_embeds * obj_masks.unsqueeze(-1).to(obj_embeds.dtype)
        
#         # Apply dropout
#         #obj_embeds = self.dropout(obj_embeds)

#         # Semantic classification head
#         obj_sem_cls = None
#         if self.sem_head is not None:
#             obj_sem_cls = self.sem_head(obj_embeds)
#             # Optional: also mask semantic logits if your loss doesn't handle it
#             # obj_sem_cls = obj_sem_cls * obj_masks.unsqueeze(-1).to(obj_sem_cls.dtype)
        
#         return obj_embeds, obj_sem_cls
