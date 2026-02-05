import os
import sys
import torch
from omegaconf import OmegaConf

# keep your sys.path tweak
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.msr3d.msr3d import MSR3D
from data.datasets.msr3d import MSR3DBase, MSQAScanNet


class MSR3DInteractiveService:
    """
    Load MSR3D once and answer (scene_id, question, situation) repeatedly.
    """
    def __init__(self, experiment_path: str, split: str = "test", device: str | None = None):
        self.experiment_path = experiment_path
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # config
        cfg_path = os.path.join(experiment_path, "config.yaml")
        if not os.path.exists(cfg_path):
            # fallback to your hardcoded path if you insist
            cfg_path = "MSR3D_BLIPT_PTv3_VIC_LORA_2/config.yaml"
        self.cfg = OmegaConf.load(cfg_path)

        # dataset (used only to build base_sample via __getitem__)
        self.dataset = MSQAScanNet(self.cfg, split=split)

        # model
        self.model = MSR3D(self.cfg).to(self.device)
        ckpt_path = os.path.join(experiment_path, "best.pth", "pytorch_model.bin")
        if not os.path.exists(ckpt_path):
            # your code had 'best.pth/pytorch_model.bin' (string with slash) — this is the correct join version.
            ckpt_path = os.path.join(experiment_path, "best.pth/pytorch_model.bin")  # fallback
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("model", state)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def change_split(self, split: str):
        split = (split or "test").lower()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split: {split}")
        self.dataset = MSQAScanNet(self.cfg, split=split)

    def _index_for_scene(self, scene_id: str) -> int:
        for i, meta in enumerate(self.dataset.data):
            if meta.get("scan_id") == scene_id:
                return i
        raise ValueError(f"No samples found for scan_id={scene_id} in dataset split")

    def _broadcast_list(self, v, bs, default=""):
        if isinstance(v, list):
            if len(v) == bs:
                return v
            if len(v) == 1:
                return v * bs
            return (v * bs)[:bs]
        if isinstance(v, str):
            return [v] * bs
        if v is None:
            return [default] * bs
        return [v] * bs

    def _ensure_batched(self, data_dict, bs=1):
        for k in [
            "msr3d_prompt",
            "prompt_before_obj",
            "prompt_middle_1",
            "prompt_middle_2",
            "prompt_after_obj",
            "text_output",
            "answer_list",
        ]:
            if k in data_dict:
                default = "" if k != "answer_list" else ""
                data_dict[k] = self._broadcast_list(data_dict[k], bs, default=default)

        # img_fts
        if "img_fts" in data_dict:
            if not isinstance(data_dict["img_fts"], torch.Tensor):
                data_dict["img_fts"] = torch.tensor(data_dict["img_fts"])
            if data_dict["img_fts"].dim() == 3:
                data_dict["img_fts"] = data_dict["img_fts"].unsqueeze(0)

        # img_masks
        if "img_masks" not in data_dict or not isinstance(data_dict["img_masks"], torch.Tensor):
            has_img = (
                "img_fts" in data_dict
                and isinstance(data_dict["img_fts"], torch.Tensor)
                and data_dict["img_fts"].shape[0] >= 1
            )
            val = 1 if has_img else 0
            data_dict["img_masks"] = torch.full((bs, 1), bool(val), dtype=torch.bool)
        else:
            m = data_dict["img_masks"]
            if m.dim() == 0:
                m = m.view(1, 1)
            elif m.dim() == 1:
                m = m.view(-1, 1)
            elif m.dim() > 2:
                m = m.view(m.shape[0], -1)[:, :1]
            data_dict["img_masks"] = m.to(torch.bool)

        # msr3d_img_masks
        if "msr3d_img_masks" in data_dict:
            m = data_dict["msr3d_img_masks"]
            if not isinstance(m, torch.Tensor):
                m = torch.tensor(m)
            if m.dim() == 0:
                m = m.view(1, 1)
            elif m.dim() == 1:
                m = m.view(-1, 1)
            elif m.dim() > 2:
                m = m.view(m.shape[0], -1)[:, :1]
            data_dict["msr3d_img_masks"] = m.to(torch.bool)
        else:
            data_dict["msr3d_img_masks"] = data_dict["img_masks"].clone()

        # obj_fts
        if "obj_fts" in data_dict:
            if not isinstance(data_dict["obj_fts"], torch.Tensor):
                data_dict["obj_fts"] = torch.tensor(data_dict["obj_fts"])
            if data_dict["obj_fts"].dim() == 3:
                data_dict["obj_fts"] = data_dict["obj_fts"].unsqueeze(0)
            data_dict["obj_fts"] = data_dict["obj_fts"].float()

        # obj_locs
        if "obj_locs" in data_dict:
            if not isinstance(data_dict["obj_locs"], torch.Tensor):
                data_dict["obj_locs"] = torch.tensor(data_dict["obj_locs"])
            if data_dict["obj_locs"].dim() == 2:
                data_dict["obj_locs"] = data_dict["obj_locs"].unsqueeze(0)
            data_dict["obj_locs"] = data_dict["obj_locs"].float()

        # anchors
        data_dict.setdefault("anchor_orientation", torch.zeros(4).float())
        data_dict.setdefault("anchor_locs", torch.zeros(3).float())

        if not isinstance(data_dict["anchor_orientation"], torch.Tensor):
            data_dict["anchor_orientation"] = torch.tensor(data_dict["anchor_orientation"]).float()
        if data_dict["anchor_orientation"].dim() == 1:
            data_dict["anchor_orientation"] = data_dict["anchor_orientation"].unsqueeze(0)

        if not isinstance(data_dict["anchor_locs"], torch.Tensor):
            data_dict["anchor_locs"] = torch.tensor(data_dict["anchor_locs"]).float()
        if data_dict["anchor_locs"].dim() == 1:
            data_dict["anchor_locs"] = data_dict["anchor_locs"].unsqueeze(0)

        data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
        return data_dict

    def _to_device(self, x):
        if torch.is_tensor(x):
            return x.to(self.device)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self._to_device(v) for v in x)
        return x

    def _compose_sample(self, base_sample, question: str, situation: str, images=None):
        images = images or []

        prompt = MSR3DBase.get_text_prompts(instruction=question, situation=situation)
        prompt, _ = MSR3DBase.parse_place_holder(prompt)

        has_imgs = isinstance(images, list) and len(images) > 0
        img_fts = torch.stack(images) if has_imgs else torch.zeros(3, 224, 224)
        img_masks = torch.BoolTensor([1] * len(images)) if has_imgs else torch.BoolTensor([0])
        img_masks_2d = torch.ones(1, 1, dtype=torch.bool) if has_imgs else torch.zeros(1, 1, dtype=torch.bool)

        data_dict = {
            "obj_fts": base_sample["obj_fts"],
            "obj_locs": base_sample["obj_locs"],
            "anchor_locs": base_sample["anchor_locs"],
            "anchor_orientation": base_sample["anchor_orientation"],
            "scan_id": base_sample["scan_id"],
            "img_fts": img_fts,
            "img_masks": img_masks,
            "msr3d_prompt": prompt,
            "msr3d_imgs": images if has_imgs else [],
            "msr3d_img_masks": img_masks_2d,
            "text_output": "",
            "answer_list": "",
            "source": "gradio_chat",
            "prompt_before_obj": "",
            "prompt_middle_1": "",
            "prompt_middle_2": "",
            "prompt_after_obj": "",
            "index": -1,
            "type": "custom",
        }

        # This line in your original code assumes cfg.data.msqa_scannet.args.max_obj_len exists.
        max_obj_len = self.cfg.data.msqa_scannet.args.max_obj_len
        data_dict["obj_masks"] = (torch.arange(max_obj_len) < len(data_dict["obj_locs"])).unsqueeze(0)

        data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
        return data_dict

    def answer(self, scene_id: str, question: str, situation: str) -> str:
        idx = self._index_for_scene(scene_id)
        base_sample = self.dataset[idx]
        data_dict = self._compose_sample(base_sample, question=question, situation=situation, images=[])
        data_dict = self._ensure_batched(data_dict, bs=1)
        data_dict = self._to_device(data_dict)

        with torch.no_grad():
            out = self.model.generate(data_dict)

        text = self.model.llm_tokenizer.batch_decode(out["output_tokens"], skip_special_tokens=True)
        return text[0] if text else "No answer generated."
