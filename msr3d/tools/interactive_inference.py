import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.msr3d.msr3d import MSR3D
from data.datasets.scannet_base import ScanNetBase
from data.datasets.msr3d import MSR3DBase
class InteractiveInferenceTool:
   """Tool for interactive inference using a pre-trained MSR3D model.
   
   Attributes:
      model: The MSR3D model for inference.
      cfg: Configuration settings for the model.
   """
   def __init__(self, situation, question):
      experiment_path = '/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/MSR3D_BLIP_PNPP_ViC_LORA_TUNED'   
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.cfg = self.load_config(experiment_path)
      self.model = self.load_model(os.path.join(experiment_path,'best.pth'))     
      self.data_loader = ScanNetBase(self.cfg, split='val')
      self.data_dict = self.load_data('scene0090_00')  
      self.data_dict = self.process_custom_input(question, situation, [])
      self.data_dict = self._ensure_batched(self.data_dict, bs=1)  
   def to_device(self, data, device):
        # move any torch.Tensor in dict (or nested dict/list) to device
        if torch.is_tensor(data):
            return data.to(device)
        if isinstance(data, dict):
            return {k: self.to_device(v, device) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self.to_device(v, device) for v in data)
        return data
   def _broadcast_list(self, v, bs, default=''):
        """Make sure v is a list of length bs."""
        if isinstance(v, list):
            if len(v) == bs: return v
            if len(v) == 1:  return v * bs
            # fallback: trim or pad
            return (v * bs)[:bs]
        if isinstance(v, str):
            return [v] * bs
        if v is None:
            return [default] * bs
        # already tensor or other type → wrap
        return [v] * bs

   def _ensure_batched(self, data_dict, bs=1):
      """Normalize fields to batch format the model expects."""
      # 1) prompts and text fields as lists
      for k in ['msr3d_prompt', 'prompt_before_obj', 'prompt_middle_1',
               'prompt_middle_2', 'prompt_after_obj', 'text_output', 'answer_list']:
         if k in data_dict:
               default = '' if k != 'answer_list' else ''
               data_dict[k] = self._broadcast_list(data_dict[k], bs, default=default)

      # 2) image features: (B,3,H,W)
      if 'img_fts' in data_dict:
         if isinstance(data_dict['img_fts'], torch.Tensor):
               if data_dict['img_fts'].dim() == 3:  # (3,H,W) -> (1,3,H,W)
                  data_dict['img_fts'] = data_dict['img_fts'].unsqueeze(0)
         else:
               # if it was a numpy array etc., coerce to tensor
               data_dict['img_fts'] = torch.tensor(data_dict['img_fts'])
               if data_dict['img_fts'].dim() == 3:
                  data_dict['img_fts'] = data_dict['img_fts'].unsqueeze(0)

      # 3) image masks: (B,1) bool
      if 'img_masks' not in data_dict or not isinstance(data_dict['img_masks'], torch.Tensor):
         # if no images, keep as all False; else True
         has_img = ('img_fts' in data_dict and isinstance(data_dict['img_fts'], torch.Tensor) 
                     and data_dict['img_fts'].shape[0] >= 1)
         val = 1 if has_img else 0
         data_dict['img_masks'] = torch.full((bs, 1), bool(val), dtype=torch.bool)
      else:
         m = data_dict['img_masks']
         if m.dim() == 1:                      # (B,) -> (B,1)
               data_dict['img_masks'] = m.view(-1, 1).to(torch.bool)
         elif m.dim() == 2:
               data_dict['img_masks'] = m.to(torch.bool)
         else:
               data_dict['img_masks'] = m.reshape(bs, 1).to(torch.bool)

      # 4) anchors: make sure tensors exist and have right shapes
      data_dict.setdefault('anchor_orientation', torch.zeros(4).float())
      data_dict.setdefault('anchor_locs', torch.zeros(3).float())

      # 5) final pass to fill any remaining required keys
      data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
      return data_dict    
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
      self.data_dict = self.to_device(self.data_dict, self.device)
      with torch.no_grad():
         output_dict = self.model.generate(self.data_dict)
      return output_dict

   def process_custom_input(self, question, situation, images):
      """
      Process custom inputs to generate a data dictionary for inference.

      Args:
         question (str): The question to ask the model.
         situation (str): The situation description.
         images (list): List of image tensors.

      Returns:
         dict: Processed data dictionary ready for inference.
      """
      # Generate the prompt
      prompt = MSR3DBase.get_text_prompts(instruction=question, situation=situation)
      _, place_holder_list = MSR3DBase.parse_place_holder(prompt)
      if images:
        # images: list of (3,H,W) tensors
        img_masks = torch.ones(img_fts.size(0), 1, dtype=torch.bool)  # (B,1)
      else:
        img_masks = torch.zeros(1, 1, dtype=torch.bool)               # (B,1), all False
      # Prepare the data dictionary
      data_dict = {
         'source': 'custom_input',
         'scan_id': '',  # No scan ID for custom input
         'obj_fts': torch.zeros(len(images), 3, 224, 224),  # Placeholder for object features
         'obj_locs': torch.zeros(len(images), 6),  # Placeholder for object locations
         'img_fts': torch.stack(images) if images else torch.zeros(3, 224, 224),
         'img_masks': torch.BoolTensor([1] * len(images)) if images else torch.BoolTensor([0]),
         'text_output': '',  # Placeholder for text output
         'answer_list': '',  # Placeholder for answer list
         'msr3d_prompt': prompt,
         'msr3d_imgs': images,
         'anchor_orientation': torch.zeros(4).float(),
         'anchor_locs': torch.zeros(3).float(),
         'index': -1,  # Custom input index
         'type': 'custom',
         'msr3d_img_masks': img_masks
         
      }
      # Ensure all required keys are present
      data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
      return data_dict

def main():   
    # Example usage: Perform inference on a specific scene and question
    #scene_id = input("Enter scene ID (e.g., scene0090_00): ")
    #question = input("Enter your question: ")
    scene_id = "scene0000_00"
    question = "What is the color of the office chair in front of me?"    
    situation = "To my left, at a middle distance, there's a gray fabric office chair with a curved rectangle shape. Far in front, there's a gray plastic bin. Far behind, there's a crumpled red pillow and a partly open grey curtain. Near to my right, there's a black and brown fabric office chair."
    # Load scene data dynamically
    tool = InteractiveInferenceTool(situation, question)
    print("InteractiveInferenceTool initialized.")
       
    # Perform forward pass
    output_dict = tool.forward()
    
    # Decode and print the answer
    answer = tool.model.llm_tokenizer.batch_decode(output_dict['output_tokens'], skip_special_tokens=True)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
