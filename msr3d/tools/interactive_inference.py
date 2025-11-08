import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.msr3d.msr3d import MSR3D
from data.datasets.scannet_base import ScanNetBase
from data.datasets.msr3d import MSR3DBase  # your MSR3DBase from the dataset code

class InteractiveInferenceTool:
    """
    Tool for interactive inference using a pre-trained MSR3D model on ONE scene.
    Mirrors MSQAScanNet.__getitem__ but for a single (scene, question, situation).
    """
    def __init__(self, scene_id: str, situation: str, question: str, insts=None, images=None):
        experiment_path = '/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/MSR3D_BLIP_PNPP_ViC_LORA_TUNED'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = self.load_config(experiment_path)
        self.model = self.load_model(experiment_path)  # <— pass dir, not a file
        self.split = 'val'

        # dataset helpers
        self.scan_loader = ScanNetBase(self.cfg, split=self.split)  # provides _load_one_scan
        self.scene_id = scene_id
        self.situation = situation
        self.question = question
        self.insts = insts or []        # list of instance ids to prioritize; can be []
        self.images = images or []      # list of (3,H,W) tensors if you really have images

        # build one data_dict
        self.data_dict = self._build_single_datadict()
        self.data_dict = self._ensure_batched(self.data_dict, bs=1)

    # ---------- I/O ----------
    def load_config(self, exp_dir):
        return OmegaConf.load(os.path.join(exp_dir, 'config.yaml'))

    def load_model(self, exp_dir):
        model = MSR3D(self.cfg).to(self.device)
        # Expect a standard HF-style snapshot folder with pytorch_model.bin
        ckpt_path = os.path.join(exp_dir, 'pytorch_model.bin')
        if not os.path.isfile(ckpt_path):
            # fall back to best.pth if that’s your format
            alt = os.path.join(exp_dir, 'best.pth')
            if os.path.isfile(alt):
                ckpt_path = alt
        state = torch.load(ckpt_path, map_location='cpu')
        # handle DDP/non-DDP keys
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
        model.eval()
        return model

    # ---------- utils ----------
    def to_device(self, data, device):
        if torch.is_tensor(data):
            return data.to(device)
        if isinstance(data, dict):
            return {k: self.to_device(v, device) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self.to_device(v, device) for v in data)
        return data

    def _broadcast_list(self, v, bs, default=''):
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
        # 1) prompts and text fields
        for k in ['msr3d_prompt','prompt_before_obj','prompt_middle_1',
                  'prompt_middle_2','prompt_after_obj','text_output','answer_list']:
            if k in data_dict:
                default = '' if k != 'answer_list' else ''
                data_dict[k] = self._broadcast_list(data_dict[k], bs, default=default)

        # 2) img_fts (B,3,H,W)
        if 'img_fts' in data_dict:
            x = data_dict['img_fts']
            if isinstance(x, torch.Tensor):
                if x.dim() == 3:
                    x = x.unsqueeze(0)
            else:
                x = torch.tensor(x)
                if x.dim() == 3:
                    x = x.unsqueeze(0)
            data_dict['img_fts'] = x

        # 3) img_masks (B,1) bool
        if 'img_masks' not in data_dict or not isinstance(data_dict['img_masks'], torch.Tensor):
            has_img = ('img_fts' in data_dict and isinstance(data_dict['img_fts'], torch.Tensor)
                       and data_dict['img_fts'].shape[0] >= 1)
            val = 1 if has_img else 0
            data_dict['img_masks'] = torch.full((bs,1), bool(val), dtype=torch.bool)
        else:
            m = data_dict['img_masks']
            if m.dim() == 1:
                m = m.view(-1,1)
            elif m.dim() > 2:
                m = m.reshape(bs,1)
            data_dict['img_masks'] = m.to(torch.bool)

        # anchors (ensure existence)
        data_dict.setdefault('anchor_orientation', torch.zeros(4).float())
        data_dict.setdefault('anchor_locs', torch.zeros(3).float())

        # final required keys check
        data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
        return data_dict

    # ---------- core builders ----------
    def _load_scan_data(self):
        # Use your ScanNetBase helper to get per-scene data
        scan_id, scan_data = self.scan_loader._load_one_scan(
            self.scene_id, pc_type='gt', load_inst_info=True, load_pc_info=True
        )
        return scan_data

    def _scene_encoder_from_scan(self, scan_data, insts, anchor_loc=None, anchor_quat=None):
        """
        Mirror MSR3DBase._get_scene_encoder_input + preprocess_pcd
        """
        # We need access to MSR3DBase params: num_points, max_obj_len, use_rotate, split
        # Create a tiny shim object to reuse its logic cleanly.
        class _Shim(MSR3DBase):
            def __init__(self, cfg, split):
                super().__init__(cfg, dataset='ScanNet')
                self.split = split
                dcfg = cfg.data.msqa_scannet.args
                self.num_points = dcfg.get('num_points', 1024)
                self.max_obj_len = dcfg.get('max_obj_len', 60)
                self.use_rotate = dcfg.get('use_rotate', True)
                self.use_rotate = self.use_rotate and (split == 'train')

        shim = _Shim(self.cfg, self.split)

        # Build the same structure expected by _get_scene_encoder_input
        scan_data_like = {'obj_pcds': scan_data['obj_pcds']}
        situation = None
        if anchor_loc is not None and anchor_quat is not None:
            situation = (anchor_loc, anchor_quat)

        out = shim._get_scene_encoder_input(scan_data_like, insts, situation=situation)
        return out  # contains obj_fts, obj_locs, and maybe "situation"

    def _build_single_datadict(self):
        # 1) prompt + placeholders
        prompt = MSR3DBase.get_text_prompts(instruction=self.question, situation=self.situation)
        prompt_resolved, placeholder_list = MSR3DBase.parse_place_holder(prompt)

        # 2) load scan + compute obj_fts/obj_locs like in __getitem__
        scan_data = self._load_scan_data()
        scene_out = self._scene_encoder_from_scan(scan_data, self.insts)
        obj_fts = scene_out['obj_fts']
        obj_locs = scene_out['obj_locs']

        # 3) images & placeholders (keep images off unless you truly have them)
        img_list = []
        if self.images:  # user-provided list of (3,H,W) tensors
            # we won’t inject them into prompt; we’ll just pass them through
            try:
                img_fts = torch.stack(self.images)  # (B,3,H,W) expected later
                img_masks = torch.ones((img_fts.shape[0],1), dtype=torch.bool)
            except:
                img_fts = torch.zeros(3,224,224)   # will become (1,3,224,224)
                img_masks = torch.zeros(1,1, dtype=torch.bool)
        else:
            img_fts = torch.zeros(3,224,224)
            img_masks = torch.zeros(1,1, dtype=torch.bool)

        # If prompt had IMG placeholders but we didn’t resolve any, replace all with text form:
        if "IMG" in prompt and len(img_list) == 0:
            prompt_resolved = MSR3DBase.replace_all_imgs_with_txt(self=MSR3DBase, data=prompt)

        # 4) pack dict consistent with MSQAScanNet.__getitem__
        data_dict = {
            'source': 'custom_input',
            'scan_id': self.scene_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,

            'img_fts': img_fts,
            'img_masks': img_masks,

            'text_output': '',                      # unknown target at inference
            'answer_list': '',                      # not used at inference
            'msr3d_prompt': prompt_resolved,
            'msr3d_imgs': img_list,

            'anchor_orientation': torch.zeros(4).float(),  # leave zeros unless you compute it
            'anchor_locs': torch.zeros(3).float(),         # same
            'prompt_before_obj': '',
            'prompt_middle_1' : '',
            'prompt_middle_2' : '',
            'prompt_after_obj' : '',
            'index': -1,
            'type': 'custom'
        }
        return data_dict

    # ---------- inference ----------
    def forward(self):
        inputs = self.to_device(self.data_dict, self.device)
        with torch.no_grad():
            out = self.model.generate(inputs)
        return out

def main():
    scene_id = "scene0000_00"
    question = "What is the color of the office chair in front of me?"
    situation = ("To my left, at a middle distance, there's a gray fabric office chair with a curved rectangle shape. "
                 "Far in front, there's a gray plastic bin. Far behind, there's a crumpled red pillow and a partly open grey curtain. "
                 "Near to my right, there's a black and brown fabric office chair.")

    tool = InteractiveInferenceTool(scene_id=scene_id, situation=situation, question=question, insts=[])
    print("InteractiveInferenceTool initialized.")

    output = tool.forward()
    answer = tool.model.llm_tokenizer.batch_decode(output['output_tokens'], skip_special_tokens=True)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
