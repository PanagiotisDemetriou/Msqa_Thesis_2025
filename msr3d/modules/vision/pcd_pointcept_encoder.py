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
#         """
#         B, O, P, C = obj_pcds.shape
#         device = obj_pcds.device


#         point_data = transform_obj_pcds_to_pointcept(
#             obj_pcds=obj_pcds,
#             grid_size=self.grid_size,
#         )
#         point_data = move_pointcept_data_to_device(point_data, device)

#         core = self._get_core().to(device).eval() if self.freeze else self._get_core().to(device)

#         if self.freeze:
#             with torch.no_grad():
#                 point_out = core.backbone(point_data)
#         else:
#             point_out = core.backbone(point_data)

#         obj_feats = pool_point_features_to_objects(
#             point_feats=point_out.feat,
#             obj_id=point_data["obj_id"],
#             num_objs=B * O,
#             reduce=self.feat_reduce,
#         )


#         obj_feats = self.dropout(obj_feats)

#         obj_embeds = obj_feats.view(B, O, -1)

#         obj_sem_cls = None
#         if self.sem_head is not None:
#             obj_sem_cls = self.sem_head(obj_embeds)
#         return obj_embeds, obj_sem_cls

import torch
from torch import nn
import torch_scatter
from collections import OrderedDict
from pointcept.utils.config import Config as PCConfig
from pointcept.models import build_model
from pointcept.models.utils import batch2offset

from modules.build import VISION_REGISTRY
from modules.utils import get_mlp_head

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
        if ckpt_has_module and not model_has_module and kk.startswith("module."):
            kk = kk[7:]
        elif (not ckpt_has_module) and model_has_module:
            kk = "module." + kk
        weight[kk] = v

    missing, unexpected = model.load_state_dict(weight, strict=strict)
    print(f"[PTv3 Checkpoint] Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    return model

def _scatter_reduce(point_feats, index, dim_size, reduce="mean"):
    return torch_scatter.scatter(
        src=point_feats,
        index=index.long(),
        dim=0,
        dim_size=int(dim_size),
        reduce=reduce,
    )

@VISION_REGISTRY.register()
class PTv3PcdObjEncoder(nn.Module):
    """
    Scene-level PTv3 encoder (Pointcept) that pools per-point features to per-object embeddings.

    Inputs:
      scene_pcd: list[Tensor(Ni,9)] or Tensor(B,Ni,9) padded (discouraged) or Tensor(N,9) single
      instance_ids: list[Tensor(Ni,)] or Tensor matching scene_pcd points
        - per point instance index in [0..Ki-1], with -1 allowed (ignored)
    Returns:
      obj_embeds: (B, Omax, F)
      obj_masks:  (B, Omax) True=valid
      obj_sem_cls: optional (B,Omax,num_classes)
    """
    def __init__(
        self,
        cfg,
        embedding_size: int,
        ptv3_cfg_path: str,
        weight_path: str = None,
        grid_size: float = 0.02,
        feat_reduce: str = "mean",
        sem_num_classes: int = None,
        dropout: float = 0.1,
        freeze: bool = False,
        rgb_in_cols_3_6: bool = True,
        normals_in_cols_6_9: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.grid_size = float(grid_size)
        self.feat_reduce = feat_reduce
        self.freeze = freeze
        self.embedding_size = int(embedding_size)
        self.rgb_in_cols_3_6 = rgb_in_cols_3_6
        self.normals_in_cols_6_9 = normals_in_cols_6_9

        self.ptv3_cfg = PCConfig.fromfile(ptv3_cfg_path)
        model = build_model(self.ptv3_cfg.model)
        if weight_path is not None:
            model = load_pointcept_checkpoint(model, weight_path, strict=False)
        self.model = model
        self.dropout = nn.Dropout(dropout)

        self.sem_num_classes = sem_num_classes
        if sem_num_classes is not None:
            self.sem_head = get_mlp_head(self.embedding_size, 384, int(sem_num_classes), dropout=0.3)
        else:
            self.sem_head = None

        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _get_core(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _to_list(self, x):
        # Accept list[Tensor] or Tensor for single/batched
        if isinstance(x, list):
            return x
        if torch.is_tensor(x):
            if x.dim() == 2:
                return [x]
            if x.dim() == 3:
                # (B, N, C) => list of B tensors; assumes already unpadded (rare)
                return [x[b] for b in range(x.shape[0])]
        raise TypeError("Expected list[Tensor] or Tensor with dim 2/3")

    def _transform_scene_to_pointcept(self, scene_list, inst_list):
        """
        Build Pointcept dict from a batch of scenes.
        """
        assert len(scene_list) == len(inst_list)
        B = len(scene_list)

        coords = []
        feats = []
        batch = []
        inst_all = []
        num_points_per_scene = []

        for b in range(B):
            pcd = scene_list[b]
            inst = inst_list[b]

            if pcd.ndim != 2 or pcd.shape[1] < 3:
                raise ValueError(f"scene_pcd[{b}] must be (N,C) with C>=3, got {tuple(pcd.shape)}")
            if inst.ndim != 1 or inst.shape[0] != pcd.shape[0]:
                raise ValueError(f"instance_ids[{b}] must be (N,), got {tuple(inst.shape)} vs N={pcd.shape[0]}")

            xyz = pcd[:, :3]
            feat_parts = []

            if self.rgb_in_cols_3_6 and pcd.shape[1] >= 6:
                rgb = pcd[:, 3:6]
                # normalize to [0,1] if needed (handles [-1,1] or [0,255])
                rgb_min = float(rgb.min())
                rgb_max = float(rgb.max())
                if rgb_min < 0.0:          # [-1,1]
                    rgb = (rgb + 1.0) * 0.5
                elif rgb_max > 1.5:        # [0,255]
                    rgb = rgb / 255.0
                rgb = rgb.clamp(0.0, 1.0)
                feat_parts.append(rgb)

            if self.normals_in_cols_6_9 and pcd.shape[1] >= 9:
                nrm = pcd[:, 6:9]
                nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-6)
                feat_parts.append(nrm)

            feat = torch.cat(feat_parts, dim=1) if len(feat_parts) else torch.zeros((xyz.shape[0], 1), device=pcd.device)

            coords.append(xyz)
            feats.append(feat)
            batch.append(torch.full((xyz.shape[0],), b, device=pcd.device, dtype=torch.long))
            inst_all.append(inst.long())
            num_points_per_scene.append(int(xyz.shape[0]))

        coord = torch.cat(coords, dim=0)
        feat = torch.cat(feats, dim=0)
        batch = torch.cat(batch, dim=0)
        inst_all = torch.cat(inst_all, dim=0)

        offset = batch2offset(batch)

        point_data = {
            "coord": coord,
            "feat": feat,
            "batch": batch,
            "offset": offset,
            "grid_size": float(self.grid_size),
            "condition": ["ScanNet"],
        }
        return point_data, inst_all, num_points_per_scene

    def forward(self, scene_pcd, instance_ids, **kwargs):
        device = scene_pcd[0].device if isinstance(scene_pcd, list) else scene_pcd.device

        scene_list = self._to_list(scene_pcd)
        inst_list = self._to_list(instance_ids)

        # Ensure tensors on same device
        scene_list = [x.to(device) for x in scene_list]
        inst_list = [x.to(device) for x in inst_list]

        point_data, inst_all, npts_per_scene = self._transform_scene_to_pointcept(scene_list, inst_list)
        point_data = move_pointcept_data_to_device(point_data, device)

        core = self._get_core().to(device)
        if self.freeze:
            core.eval()
            with torch.no_grad():
                point_out = core.backbone(point_data)
        else:
            point_out = core.backbone(point_data)

        point_feats = point_out.feat  # (N_total, F)

        # ---- pool to objects ----
        # We need a global object id per point across the batch.
        # For each scene b: instance ids in [0..Kb-1]. We offset them by cumulative K.
        # Ignore inst == -1 points.
        batch = point_data["batch"]  # (N_total,)
        valid = inst_all >= 0
        if not valid.any():
            # no valid instances
            B = len(scene_list)
            obj_embeds = point_feats.new_zeros((B, 0, point_feats.shape[1]))
            obj_masks = torch.zeros((B, 0), device=device, dtype=torch.bool)
            obj_sem_cls = self.sem_head(obj_embeds) if self.sem_head is not None else None
            return obj_embeds, obj_sem_cls, obj_masks

        inst_v = inst_all[valid]
        batch_v = batch[valid]
        feats_v = point_feats[valid]

        # compute K per scene (max inst + 1), then offsets
        B = len(scene_list)
        K_per_scene = []
        for b in range(B):
            inst_b = inst_list[b]
            vb = inst_b[inst_b >= 0]
            Kb = int(vb.max().item()) + 1 if vb.numel() > 0 else 0
            K_per_scene.append(Kb)

        obj_base = torch.zeros((B,), device=device, dtype=torch.long)
        running = 0
        for b in range(B):
            obj_base[b] = running
            running += K_per_scene[b]
        total_objs = int(running)

        global_obj_id = inst_v + obj_base[batch_v]  # (N_valid,)

        obj_feats = _scatter_reduce(feats_v, global_obj_id, dim_size=total_objs, reduce=self.feat_reduce)
        obj_feats = self.dropout(obj_feats)  # (total_objs, F)

        # ---- pad back to (B, Omax, F) + mask ----
        Omax = max(K_per_scene) if len(K_per_scene) else 0
        Fdim = obj_feats.shape[1]
        obj_embeds = obj_feats.new_zeros((B, Omax, Fdim))
        obj_masks = torch.zeros((B, Omax), device=device, dtype=torch.bool)

        start = 0
        for b in range(B):
            Kb = K_per_scene[b]
            if Kb > 0:
                obj_embeds[b, :Kb] = obj_feats[start:start+Kb]
                obj_masks[b, :Kb] = True
            start += Kb

        obj_sem_cls = self.sem_head(obj_embeds) if self.sem_head is not None else None
        return obj_embeds, obj_sem_cls, obj_masks
