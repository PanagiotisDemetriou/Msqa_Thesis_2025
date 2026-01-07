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
    print(f"[PTv3 Checkpoint] Missing: {len(missing)}  Unexpected: {len(unexpected)}")
    return model


def transform_obj_pcds_to_pointcept(obj_pcds, grid_size=0.02, rgb_div=255.0):
    """
    obj_pcds: (B, O, P, 9) with [xyz(3), rgb(3), normals(3)]
             rgb expected in 0..255 range (will be scaled to 0..1).
    returns point_data dict for PTv3 backbone.
    """
    if obj_pcds.ndim != 4:
        raise ValueError(f"obj_pcds must be (B,O,P,C). Got {tuple(obj_pcds.shape)}")

    B, O, P, C = obj_pcds.shape
    if C < 9:
        raise ValueError(f"Expected at least 9 channels [xyz,rgb,normals]. Got C={C}")

    total_points = B * O * P

    coord = obj_pcds[..., :3].reshape(total_points, 3)

    rgb = obj_pcds[..., 3:6].reshape(total_points, 3) / float(rgb_div)
    normals = obj_pcds[..., 6:9].reshape(total_points, 3)

    # Pointcept feature: concatenate rgb + normals -> (N_total, 6)
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


        obj_feats = self.dropout(obj_feats)

        obj_embeds = obj_feats.view(B, O, -1)

        obj_sem_cls = None
        if self.sem_head is not None:
            obj_sem_cls = self.sem_head(obj_embeds)
        return obj_embeds, obj_sem_cls

