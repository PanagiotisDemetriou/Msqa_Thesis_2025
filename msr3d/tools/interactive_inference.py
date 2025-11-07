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
      print("InteractiveInferenceTool initialized.")
      self.data_dict = self.process_custom_input(question, situation, [])
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
    
    # Batch size (1 if you’re doing single-sample inference)
    bs = 1

    # 1) Provide a list[str] for msr3d_prompt (not a plain str)
    prompt_str = MSR3DBase.get_text_prompts(instruction=question, situation=situation)
    scan_data['msr3d_prompt'] = [prompt_str] * bs

    # 2) Make sure the LEO-style pieces are lists if present (or let the helper fill them)
    for k in ['prompt_before_obj','prompt_middle_1','prompt_middle_2','prompt_after_obj','text_output','answer_list']:
       v = scan_data.get(k)
       if isinstance(v, str):
          scan_data[k] = [v] * bs
       elif v is None:
          # minimal blanks the model can accept
          scan_data[k] = [''] * bs

    # 3) Ensure image tensors & masks have batch dims and correct types
    if scan_data.get('img_fts') is not None:
       if scan_data['img_fts'].dim() == 3:           # (3, H, W) -> (1, 3, H, W)
          scan_data['img_fts'] = scan_data['img_fts'].unsqueeze(0)
       # Boolean mask (B, 1) with True meaning “not masked”
       scan_data['img_masks'] = torch.ones(bs, 1, dtype=torch.bool)

    # 4) Let their utility fill any missing required keys consistently
    scan_data = MSR3DBase.check_output_and_fill_dummy(scan_data)

    tool.data_dict = scan_data
    
    
    # Perform forward pass
    output_dict = tool.forward()
    
    # Decode and print the answer
    answer = tool.model.llm_tokenizer.batch_decode(output_dict['output_tokens'], skip_special_tokens=True)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
