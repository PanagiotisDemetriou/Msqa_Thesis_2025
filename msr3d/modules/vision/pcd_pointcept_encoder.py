import einops
from omegaconf import OmegaConf
import torch
from torch import nn
import torch_scatter
from collections import OrderedDict
from triton import Config

from modules.build import VISION_REGISTRY # XXXX # was modules, the one below as well
from modules.utils import get_mlp_head

from pointcept.utils.config import Config as PCConfig
from pointcept.models import build_model
from pointcept.models.utils import batch2offset
import pointcept.utils.comm as comm
from data.datasets.ptv3_data_processing import PTV3DataProcessing

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
        # self.batch_size = cfg.dataloader.train.batchsize
        self.grid_size = float(grid_size)
        self.freeze = freeze
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Build Pointcept model from config file
        self.ptv3_cfg = PCConfig.fromfile(ptv3_cfg_path)
        model = build_model(self.ptv3_cfg.model)
        #model = build_model(self.ptv3_cfg.model.backbone)

        self.ptv3_processor = PTV3DataProcessing(self.cfg)
       
        if weight_path is not None:
            model = load_pointcept_checkpoint(model, weight_path, strict=False)

        # Keep only the backbone path you already validated
        self.model = model
       
        # Optional semantic classifier head (like your obj3d_clf_pre_head)
        self.sem_num_classes = sem_num_classes
        self.sem_head = get_mlp_head(64, 384, self.sem_num_classes, dropout=0.3).to(self.device)

        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _get_core(self):
        core = self.model.module if hasattr(self.model, "module") else self.model
        return core
 
    def forward(self, data, mode = "train"):
        obj_masks = data.get('obj_masks', None).to(self.device) if 'obj_masks' in data else None
        obj_ids = data.get('selected_obj_ids', None).to(self.device) if 'selected_obj_ids' in data else None
        offset = data['scene_offset']
        # print(data['scan_id'])
        # print("----------------")
        # print(data.keys())


        # Convert raw data to Pointcept format and move to device
        data_dict = self.ptv3_processor.create_data_dict(data)
        
        # Prepare data by applying Pointcept transforms
        data_dict = self.ptv3_processor.prepare_data(data_dict , mode)
        
        # Move data to device after processing
        data_dict = move_pointcept_data_to_device(data_dict, self.device)

        # Process through PTv3 backbone
        core = self._get_core().to(self.device).eval() if self.freeze else self._get_core().to(self.device)
        #core = self.model.to(self.device).eval()

        #print(data_dict.keys())
        if self.freeze:
            with torch.no_grad():
                point_out = core.backbone(data_dict)
                #point_out = core(data_dict)
        else:
            point_out = core.backbone(data_dict)
            #point_out = core(data_dict)

        print(len(point_out['feat']))
        print(point_out.keys())
        print(len(data_dict['inverse']))
        if "inverse" in data_dict.keys(): # XXXX # 
            assert "origin_inst" in data_dict.keys()
            point_out['feat'] = point_out['feat'][data_dict["inverse"]]
            point_out['inst_id'] = data_dict["origin_inst"]
        print(len(point_out['feat']))
        # point_out['offset'] = offset # Prepei na mpei sto data dict?
        # Pool point features to object features
        obj_embeds, obj_mask = self.ptv3_processor.pool_object_features(point_out, obj_ids) 

        
        data['obj_masks'] = obj_mask  # Add object mask to data for downstream use
        
        # print(f"Valid objects {obj_embeds[obj_mask == 1]}")
        # print(f"Padding {obj_embeds[obj_mask == 0]}")
        # Semantic classification head
        obj_sem_cls = None
        if self.sem_head is not None:
            obj_sem_cls = self.sem_head(obj_embeds)
        
        return obj_embeds, obj_sem_cls