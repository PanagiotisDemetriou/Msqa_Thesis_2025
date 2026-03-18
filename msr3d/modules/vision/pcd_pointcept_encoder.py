import einops
from omegaconf import OmegaConf
import torch
from torch import nn
import torch_scatter
from collections import OrderedDict
from triton import Config
import numpy as np
import os
os.environ["SPCONV_ALGO"] = "native"  # force Native globally before import
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
        from spconv.pytorch import ConvAlgo

        # Force Native algo to skip the broken tuner / binding crash
        for m in self.model.modules():
            if hasattr(m, 'algo'):
                print(f"Forcing Native on {m.__class__.__name__}")
                m.algo = ConvAlgo.Native
                
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

        if isinstance(offset, torch.Tensor):
            offset = offset.cpu().numpy().tolist()
        offset = [0] + offset 
        B = len(data['scan_id'])
        assert len(offset) == B + 1, f"Expected {B+1} offsets, got {len(offset)}"

        print(f"Offset: {offset}")
        print(data['scene_fts'])
        print(f"scene1:{len(data['scene_fts'])}")
        # print(f"scene2:{len(data['scene_fts'][1])}")
        # print(data['scan_id'])
        # print("----------------")
        
        # print("\n==== INSTANCE IDS CHECK ====")
        # print("unique instance_ids (first 30):", torch.unique(data['instance_ids'])[:30])
        # print("min instance_id:", data['instance_ids'].min().item())
        # print("max instance_id:", data['instance_ids'].max().item())

        # Convert raw data to Pointcept format and move to device
        data_dict = self.ptv3_processor.create_data_dict(data)

        per_scene_inputs = []

        for i in range(B):
            start = offset[i]
            end   = offset[i + 1]
            
            single = {
                'coord':   data_dict['coord'][start:end],
                'color':   data_dict['color'][start:end],
                'normal':  data_dict['normal'][start:end],
                'inst_id': data_dict['inst_id'][start:end],
                'segment': data_dict['segment'][start:end],
                'name':    data['scan_id'][i],
                
            }
            per_scene_inputs.append(single)
        processed_scenes = []
        for single_scene_dict in per_scene_inputs:
            # Apply the same transform pipeline that was used before
            processed_dict = self.ptv3_processor.prepare_data(single_scene_dict, mode)
            processed_scenes.append(processed_dict)

        data_dict = self.ptv3_processor.collate_pointcloud(processed_scenes)
        # ── DEBUG PRINTS ────────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("AFTER PER-SCENE TRANSFORM + COLLATE")
        print(f"  Number of scenes processed: {len(processed_scenes)}")
        print(f"  offset shape:               {data_dict.get('offset').shape if 'offset' in data_dict else 'missing'}")
        print(f"  offset values:              {data_dict.get('offset').tolist() if 'offset' in data_dict else 'missing'}")
        print(f"  total points (coord):       {data_dict['coord'].shape[0] if 'coord' in data_dict else 'missing'}")
        # print(f"  unique instance ids:        {np.unique(data_dict.get('inst_id', np.array([-999])))}")
        inst = data_dict.get('inst_id', torch.tensor([-999], device='cpu'))
        if isinstance(inst, torch.Tensor):
            inst = inst.cpu().numpy()   # <--- crucial: .cpu() first
        print(f" unique instance ids: {np.unique(inst)}")

        print(f"  keys in data_dict:          {sorted(data_dict.keys())}")
        print("="*60 + "\n")
        # Prepare data by applying Pointcept transforms
        #data_dict = self.ptv3_processor.prepare_data(data_dict, mode)
        
        
        # Move data to device after processing
        data_dict = move_pointcept_data_to_device(data_dict, self.device)

        # Process through PTv3 backbone
        core = self._get_core().to(self.device).eval() if self.freeze else self._get_core().to(self.device)
        #core = self.model.to(self.device).eval()

        print(f"Going into backbone with {B} scenes, {data_dict['coord'].shape[0]} points total")
        if 'offset' in data_dict:
            print("Scene point counts:", torch.diff(data_dict['offset']).tolist())
        #print(data_dict.keys())
        if self.freeze:
            with torch.no_grad(),torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                point_out = core.backbone(data_dict)
                #point_out = core(data_dict)
        else:
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                point_out = core.backbone(data_dict)
            #point_out = core(data_dict)
        if torch.isnan(point_out['feat']).any():
            print("WARNING: NaNs detected in PTv3 features")
        point_out['feat'] = torch.nan_to_num(
            point_out['feat'],
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        # print(len(point_out['feat']))
        # print(point_out.keys())
        # print(len(data_dict['inverse']))
        if "inverse" in data_dict.keys(): # XXXX # 
            assert "origin_inst" in data_dict.keys()
            point_out['feat'] = point_out['feat'][data_dict["inverse"]]
            point_out['inst_id'] = data_dict["origin_inst"]
            # Critical: use ORIGINAL offset now that we un-downsampled feat & inst_id
            if 'scene_offset' in data:  # the one you computed at the beginning
                point_out['offset'] = torch.tensor(
                    [0] + data['scene_offset'].cpu().numpy().tolist(),
                    dtype=torch.long,
                    device=data['scene_offset'].device
                )
            else:
                # fallback: reconstruct from original lengths if needed
                raise ValueError("scene_offset not found in input data")
        print("\n=== POST-INVERSE / POST-REMAPPING DEBUG ===")
        print("point_out['feat'].shape:", point_out['feat'].shape)
        print("point_out has 'inst_id':", 'inst_id' in point_out)
        if 'inst_id' in point_out:
            u = torch.unique(point_out['inst_id'])
            print("Unique inst_ids AFTER remapping:", u.tolist())
            print("Min/Max inst_id AFTER remapping:", u.min().item(), u.max().item())
            print("Number of points:", point_out['feat'].shape[0])
            print("offset in point_out?", 'offset' in point_out)
            if 'offset' in point_out:
                print("offset values:", point_out['offset'].tolist())
        # print(len(point_out['feat']))
        # point_out['offset'] = offset # Prepei na mpei sto data dict?
        # Pool point features to object features
        print(f"ID: {data['scan_id']}")
        print(f"Kept IDs:")
        print("Going to pooling with obj_ids shape:", obj_ids.shape if obj_ids is not None else "None")
        if obj_ids is not None:
            print("Scene 0 obj_ids:", obj_ids[0].tolist()[:35])
            print("Scene 1 obj_ids:", obj_ids[1].tolist()[:35])
        obj_embeds, obj_mask = self.ptv3_processor.pool_object_features(point_out, obj_ids) 
        obj_embeds = torch.nan_to_num(
            obj_embeds,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        # print("\n==== FINAL OUTPUT ====")
        # print("obj_embeds shape:", obj_embeds.shape)
        # print("obj_mask shape:", obj_mask.shape)
        # print("valid objects per scene:", obj_mask.sum(dim=1))
        
        data['obj_masks'] = obj_mask  # Add object mask to data for downstream use
        
        # print(f"Valid objects {obj_embeds[obj_mask == 1]}")
        # print(f"Padding {obj_embeds[obj_mask == 0]}")
        # Semantic classification head
        obj_sem_cls = None
        if self.sem_head is not None:
            obj_sem_cls = self.sem_head(obj_embeds)
        
        return obj_embeds, obj_sem_cls