import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.cfg = self.load_config(experiment_path)
      self.model = self.load_model(os.path.join(experiment_path,'best.pth'))
      
      self.data_loader = ScanNetBase(self.cfg, split='val')
      self.data_dict = self.load_data('scene0090_00')  
      print("InteractiveInferenceTool initialized.")
      print(self.data_dict.keys())
   def load_model(self, path):
      model = MSR3D(self.cfg)
      model = model.to(self.device)
      model_state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'))
      is_model_distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
      if is_model_distributed:
         model.module.load_state_dict(model_state_dict, strict=False)
      else:
         model.load_state_dict(model_state_dict, strict=False)
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
   def forward(self):
      """Perform inference on the input data dictionary.
      
      Args:
         data_dict: Dictionary containing input data for the model.
         
      Returns:
         output_dict: Dictionary containing model outputs.
      """
      with torch.no_grad():
         output_dict = self.model.generate(self.data_dict)
      return output_dict

def main():
    tool = InteractiveInferenceTool()
    print("InteractiveInferenceTool initialized.")
    
    # Example usage: Perform inference on a specific scene and question
    #scene_id = input("Enter scene ID (e.g., scene0090_00): ")
    #question = input("Enter your question: ")
    scene_id = "scene0090_00"
    question = "Is the bathroom stall open or closed?"
    # Load scene data dynamically
    tool.data_dict = tool.load_data(scene_id)
    
    # Add the question to the prompt
    tool.data_dict['prompt_before_obj'] = f"You are in a scene. USER: {question} ASSISTANT:"
    
    # Perform forward pass
    output_dict = tool.forward()
    
    # Decode and print the answer
    answer = tool.model.llm_tokenizer.batch_decode(output_dict['output_tokens'], skip_special_tokens=True)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
