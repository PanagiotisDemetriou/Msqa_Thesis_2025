import os
import torch
import numpy as np
from omegaconf import OmegaConf
from model.msr3d.msr3d import MSR3D
from data.datasets.scannet_base import ScanNetBase
class InteractiveInferenceTool:
   """Tool for interactive inference using a pre-trained MSR3D model.
   
   Attributes:
      model: The MSR3D model for inference.
      cfg: Configuration settings for the model.
   """
   def __init__(self):
      experiment_path = '/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/MSR3D_BLIP_PNPP_ViC_LORA_TUNED'   
      self.cfg = self.load_config(experiment_path)
      self.model = self.load_model(os.path.join(experiment_path,'best.pth'))
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.data_loader = ScanNetBase(self.cfg, split='val')
      self.data_dict = self.load_data('scene0090_00')  
      print("InteractiveInferenceTool initialized.")
      print(self.data_dict.keys())
   def load_model(self, path):
      model = MSR3D(self.cfg)
      model = model.to(self.device)
      model_state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'))
      is_model_distributed = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
      if is_model_distributed:
         self.model.module.load_state_dict(model_state_dict, strict=False)
      else:
         self.model.load_state_dict(model_state_dict, strict=False)
      model.eval()
      return model
   def load_config(self, path):
      config = OmegaConf.load(os.path.join(path,'config.yaml'))
      return config
   def load_data(self, scene_id):
      scan_id, scan_data = self.data_loader._load_one_scan(
          scene_id,
          pc_type='gt',
          load_inst_info=True,
          load_pc_info=True
      )
      return scan_data