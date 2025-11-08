import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf

# keep your sys.path tweak
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.msr3d.msr3d import MSR3D
from data.datasets.scannet_base import ScanNetBase
from data.datasets.msr3d import MSR3DBase, MSQAScanNet   # <-- use dataset to get a proper one-scene sample


class InteractiveInferenceTool:
   """Tool for interactive inference using a pre-trained MSR3D model.
   
   Attributes:
      model: The MSR3D model for inference.
      cfg: Configuration settings for the model.
   """
   def __init__(self, scene_id, situation, question):
      # DO NOT change this path (as requested)
      experiment_path = '/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/MSR3D_BLIP_PNPP_ViC_LORA_TUNED'

      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.cfg = self.load_config(experiment_path)
      self.model = self.load_model(experiment_path)  # load best.pth from the experiment dir (see load_model)
      
      # (optional) keep this around, though we now rely on the dataset to build inputs
      self.data_loader = ScanNetBase(self.cfg, split='val')

      # Build a proper sample via __getitem__ from the dataset,
      # then override the prompt with your custom (question, situation)
      self.dataset = MSQAScanNet(self.cfg, split='val')

      idx = self._index_for_scene(self.dataset, scene_id)
      base_sample = self.dataset[idx]  # this already has obj_fts/obj_locs/anchors/etc.

      # Compose the final input using your custom question+situation
      self.data_dict = self.compose_sample(base_sample, question, situation, images=[])

      # Ensure batch dimension etc.
      self.data_dict = self._ensure_batched(self.data_dict, bs=1)


   # ---------- helpers ----------
   def _index_for_scene(self, dataset, scene_id):
      """Find the first dataset index matching the given scan_id."""
      for i, meta in enumerate(dataset.data):
         if meta.get("scan_id") == scene_id:
            return i
      raise ValueError(f"No samples found for scan_id={scene_id}")

   def compose_sample(self, base_sample, question, situation, images):
      """
      Take a dataset-produced sample (via __getitem__) and override the prompt
      with your custom (question, situation). Keep geometry & anchors, etc.
      """
      # Build the new prompt and strip any placeholders
      prompt = MSR3DBase.get_text_prompts(instruction=question, situation=situation)
      prompt, _ = MSR3DBase.parse_place_holder(prompt)

      # Images handling: if you pass images, they should be (3,H,W) tensors.
      has_imgs = isinstance(images, list) and len(images) > 0
      img_fts = torch.stack(images) if has_imgs else torch.zeros(3, 224, 224)
      img_masks = torch.BoolTensor([1] * len(images)) if has_imgs else torch.BoolTensor([0])
      # Make masks 2-D (B,1) right away
      if has_imgs:
         img_masks_2d = torch.ones(1, 1, dtype=torch.bool)
      else:
         img_masks_2d = torch.zeros(1, 1, dtype=torch.bool)

      # Start from the base sample and override what we need for inference
      data_dict = {
         # keep scene geometry and pose from base sample
         'obj_fts': base_sample['obj_fts'],
         'obj_locs': base_sample['obj_locs'],
         
         'anchor_locs': base_sample['anchor_locs'],
         'anchor_orientation': base_sample['anchor_orientation'],
         'scan_id': base_sample['scan_id'],

         # vision stubs / custom images
         'img_fts': img_fts,
         'img_masks': img_masks,

         # text fields
         'msr3d_prompt': prompt,
         'msr3d_imgs': images if has_imgs else [],
         'msr3d_img_masks': img_masks_2d,

         # minimal required extras
         'text_output': '',
         'answer_list': '',
         'source': 'custom_input',
         'prompt_before_obj': '',      # filled by check_output_and_fill_dummy if needed
         'prompt_middle_1': '',
         'prompt_middle_2': '',
         'prompt_after_obj': '',
         'index': -1,
         'type': 'custom',
      }
      data_dict['obj_masks']= (torch.arange(self.cfg.data.msqa_scannet.args.max_obj_len) < len(data_dict['obj_locs'])).unsqueeze(0)
      # Ensure required keys and defaults
      data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
      return data_dict

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
         return (v * bs)[:bs]
      if isinstance(v, str):
         return [v] * bs
      if v is None:
         return [default] * bs
      return [v] * bs

   def _ensure_batched(self, data_dict, bs=1):
      # 1) prompts and text fields as lists (unchanged)
      for k in ['msr3d_prompt', 'prompt_before_obj', 'prompt_middle_1',
               'prompt_middle_2', 'prompt_after_obj', 'text_output', 'answer_list']:
         if k in data_dict:
               default = '' if k != 'answer_list' else ''
               data_dict[k] = self._broadcast_list(data_dict[k], bs, default=default)

      # 2) image features: (B,3,H,W) (unchanged)
      if 'img_fts' in data_dict:
         if not isinstance(data_dict['img_fts'], torch.Tensor):
               data_dict['img_fts'] = torch.tensor(data_dict['img_fts'])
         if data_dict['img_fts'].dim() == 3:            # (3,H,W) -> (1,3,H, W)
               data_dict['img_fts'] = data_dict['img_fts'].unsqueeze(0)

      # 3) image masks: (B,1) bool (unchanged)
      if 'img_masks' not in data_dict or not isinstance(data_dict['img_masks'], torch.Tensor):
         has_img = ('img_fts' in data_dict and isinstance(data_dict['img_fts'], torch.Tensor) 
                     and data_dict['img_fts'].shape[0] >= 1)
         val = 1 if has_img else 0
         data_dict['img_masks'] = torch.full((bs, 1), bool(val), dtype=torch.bool)
      else:
         m = data_dict['img_masks']
         if not isinstance(m, torch.Tensor):
               m = torch.tensor(m)
         if m.dim() == 0:                               # scalar -> (1,1)
               m = m.view(1, 1)
         elif m.dim() == 1:                             # (B,) -> (B,1)
               m = m.view(-1, 1)
         elif m.dim() > 2:
               m = m.view(m.shape[0], -1)[:, :1]
         data_dict['img_masks'] = m.to(torch.bool)

      # 3b) msr3d_img_masks: mirror handling so it's (B,1)
      if 'msr3d_img_masks' in data_dict:
         m = data_dict['msr3d_img_masks']
         if not isinstance(m, torch.Tensor):
               m = torch.tensor(m)
         if m.dim() == 0:
               m = m.view(1, 1)
         elif m.dim() == 1:
               m = m.view(-1, 1)
         elif m.dim() > 2:
               m = m.view(m.shape[0], -1)[:, :1]
         data_dict['msr3d_img_masks'] = m.to(torch.bool)
      else:
         data_dict['msr3d_img_masks'] = data_dict['img_masks'].clone()

      # 4) point cloud tensors — add batch dim
      # obj_fts: [num_objs, num_pts, C] -> [1, num_objs, num_pts, C]
      if 'obj_fts' in data_dict:
         if not isinstance(data_dict['obj_fts'], torch.Tensor):
               data_dict['obj_fts'] = torch.tensor(data_dict['obj_fts'])
         if data_dict['obj_fts'].dim() == 3:
               data_dict['obj_fts'] = data_dict['obj_fts'].unsqueeze(0)
         data_dict['obj_fts'] = data_dict['obj_fts'].float()

      # obj_locs: [num_objs, 6] -> [1, num_objs, 6]
      if 'obj_locs' in data_dict:
         if not isinstance(data_dict['obj_locs'], torch.Tensor):
               data_dict['obj_locs'] = torch.tensor(data_dict['obj_locs'])
         if data_dict['obj_locs'].dim() == 2:
               data_dict['obj_locs'] = data_dict['obj_locs'].unsqueeze(0)
         data_dict['obj_locs'] = data_dict['obj_locs'].float()

      # 5) anchors: [4] -> [1,4], [3] -> [1,3]
      data_dict.setdefault('anchor_orientation', torch.zeros(4).float())
      data_dict.setdefault('anchor_locs', torch.zeros(3).float())

      if not isinstance(data_dict['anchor_orientation'], torch.Tensor):
         data_dict['anchor_orientation'] = torch.tensor(data_dict['anchor_orientation']).float()
      if data_dict['anchor_orientation'].dim() == 1:
         data_dict['anchor_orientation'] = data_dict['anchor_orientation'].unsqueeze(0)

      if not isinstance(data_dict['anchor_locs'], torch.Tensor):
         data_dict['anchor_locs'] = torch.tensor(data_dict['anchor_locs']).float()
      if data_dict['anchor_locs'].dim() == 1:
         data_dict['anchor_locs'] = data_dict['anchor_locs'].unsqueeze(0)

      # 6) final pass to guarantee required keys
      data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
      return data_dict

   # ---------- model / config / scene IO ----------
   def load_model(self, experiment_path):
      """
      Load model from the experiment directory. We expect a 'best.pth' there.
      (Do not change the path; just load that file.)
      """
      model = MSR3D(self.cfg).to(self.device)
      ckpt_path = os.path.join(experiment_path, 'best.pth/pytorch_model.bin')
      state = torch.load(ckpt_path, map_location='cpu')
      # allow both raw state_dict or wrapper dicts
      state_dict = state.get('model', state)
      is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
      if is_ddp:
         model.module.load_state_dict(state_dict, strict=False)
      else:
         model.load_state_dict(state_dict, strict=False)
      model.eval()
      return model

   def load_config(self, experiment_path):
      return OmegaConf.load(os.path.join(experiment_path, 'config.yaml'))

   def load_data(self, scene_id):
      scan_id, scan_data = self.data_loader._load_one_scan(
         scene_id,
         pc_type='gt',
         load_inst_info=True,
         load_pc_info=True
      )
      return scan_data

   # ---------- inference ----------
   def forward(self):
      """Perform inference on the current self.data_dict."""
      self.data_dict = self.to_device(self.data_dict, self.device)
      with torch.no_grad():
         output_dict = self.model.generate(self.data_dict)
      return output_dict

   def debug_data_dict(self, data_dict):
      """Optional: print shapes and required keys."""
      print("\n[DEBUG] Data Dictionary:")
      for key, value in data_dict.items():
         if isinstance(value, torch.Tensor):
            print(f"{key}: Tensor with shape {value.shape}")
         else:
            print(f"{key}: {value}")
      print("\n[DEBUG] Missing Keys:")
      required_keys = [
               'msr3d_prompt',
               'msr3d_imgs',  ## (B, max_num, C, H, W) this will be padded to max_num in the dataset wrapper
               'obj_fts',
               # 'obj_masks', # this is filled by dataset wrapper
               'obj_locs',
               'img_fts',
               'img_masks',
               'text_output',
               'anchor_orientation',
               'anchor_locs',
               'source',
               'scan_id',
               'prompt_before_obj',
               'prompt_middle_1',
               'prompt_middle_2',
               'prompt_after_obj',
               'index',
               'type'
            ]
      for key in required_keys:
         if key not in data_dict:
            print(f"Missing key: {key}")

   def ask_question(self, scene_id, question, situation, images=[]):
      """
      (Re)build the sample for a given scene and custom question+situation and run inference.
      """
      # obtain a base sample from the dataset
      idx = self._index_for_scene(self.dataset, scene_id)
      base_sample = self.dataset[idx]

      # rebuild data_dict with your custom prompt and optional images
      data_dict = self.compose_sample(base_sample, question, situation, images)
      data_dict = self._ensure_batched(data_dict, bs=1)
      self.debug_data_dict(data_dict)

      # run
      data_dict = self.to_device(data_dict, self.device)
      with torch.no_grad():
         output = self.model.generate(data_dict)

      # decode
      answer = self.model.llm_tokenizer.batch_decode(output['output_tokens'], skip_special_tokens=True)
      return answer[0] if answer else "No answer generated."


def main():
   # Example usage: Perform inference on a specific scene and question
   scene_id = "scene0231_00"
   question = "What is the color of the carpet?"
   situation = ("I am sitting on the sofa and in front of me there is a trash can. To my near left there is a brown armchair and to my far right the kitchen")

   tool = InteractiveInferenceTool(scene_id, situation, question)
   print("InteractiveInferenceTool initialized.")

   # Forward once using the prebuilt data_dict
   output_dict = tool.forward()
   answer = tool.model.llm_tokenizer.batch_decode(output_dict['output_tokens'], skip_special_tokens=True)
   print("Answer (forward):", answer)

   # Or use ask_question to rebuild and run (handy if you want to change inputs)
   answer2 = tool.ask_question(scene_id, question, situation, images=[])
   print("Answer (ask_question):", answer2)


if __name__ == "__main__":
   main()
