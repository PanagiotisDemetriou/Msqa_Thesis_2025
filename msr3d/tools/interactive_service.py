# import os
# import sys
# import torch
# from omegaconf import OmegaConf

# # keep your sys.path tweak
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from model.msr3d.msr3d import MSR3D
# from data.datasets.msr3d import MSR3DBase, MSQAScanNet


# class MSR3DInteractiveService:
#     """
#     Load MSR3D once and answer (scene_id, question, situation) repeatedly.
#     """
#     def __init__(self, experiment_path: str, split: str = "test", device: str | None = None):
#         self.experiment_path = experiment_path
#         self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # config
#         cfg_path = os.path.join(experiment_path, "config.yaml")
#         if not os.path.exists(cfg_path):
#             # fallback to your hardcoded path if you insist
#             cfg_path = "MSR3D_BLIPT_PTv3_VIC_LORA_2/config.yaml"
#         self.cfg = OmegaConf.load(cfg_path)

#         # dataset (used only to build base_sample via __getitem__)
#         self.dataset = MSQAScanNet(self.cfg, split=split)

#         # model
#         self.model = MSR3D(self.cfg).to(self.device)
#         ckpt_path = os.path.join(experiment_path, "best.pth", "pytorch_model.bin")
#         if not os.path.exists(ckpt_path):
#             # your code had 'best.pth/pytorch_model.bin' (string with slash) — this is the correct join version.
#             ckpt_path = os.path.join(experiment_path, "best.pth/pytorch_model.bin")  # fallback
#         state = torch.load(ckpt_path, map_location="cpu")
#         state_dict = state.get("model", state)
#         self.model.load_state_dict(state_dict, strict=False)
#         self.model.eval()

#     def change_split(self, split: str):
#         split = (split or "test").lower()
#         if split not in ("train", "val", "test"):
#             raise ValueError(f"Unsupported split: {split}")
#         self.dataset = MSQAScanNet(self.cfg, split=split)

#     def _index_for_scene(self, scene_id: str) -> int:
#         for i, meta in enumerate(self.dataset.data):
#             if meta.get("scan_id") == scene_id:
#                 return i
#         raise ValueError(f"No samples found for scan_id={scene_id} in dataset split")

#     def _broadcast_list(self, v, bs, default=""):
#         if isinstance(v, list):
#             if len(v) == bs:
#                 return v
#             if len(v) == 1:
#                 return v * bs
#             return (v * bs)[:bs]
#         if isinstance(v, str):
#             return [v] * bs
#         if v is None:
#             return [default] * bs
#         return [v] * bs

#     def _ensure_batched(self, data_dict, bs=1):
#         for k in [
#             "msr3d_prompt",
#             "prompt_before_obj",
#             "prompt_middle_1",
#             "prompt_middle_2",
#             "prompt_after_obj",
#             "text_output",
#             "answer_list",
#         ]:
#             if k in data_dict:
#                 default = "" if k != "answer_list" else ""
#                 data_dict[k] = self._broadcast_list(data_dict[k], bs, default=default)

#         # img_fts
#         if "img_fts" in data_dict:
#             if not isinstance(data_dict["img_fts"], torch.Tensor):
#                 data_dict["img_fts"] = torch.tensor(data_dict["img_fts"])
#             if data_dict["img_fts"].dim() == 3:
#                 data_dict["img_fts"] = data_dict["img_fts"].unsqueeze(0)

#         # img_masks
#         if "img_masks" not in data_dict or not isinstance(data_dict["img_masks"], torch.Tensor):
#             has_img = (
#                 "img_fts" in data_dict
#                 and isinstance(data_dict["img_fts"], torch.Tensor)
#                 and data_dict["img_fts"].shape[0] >= 1
#             )
#             val = 1 if has_img else 0
#             data_dict["img_masks"] = torch.full((bs, 1), bool(val), dtype=torch.bool)
#         else:
#             m = data_dict["img_masks"]
#             if m.dim() == 0:
#                 m = m.view(1, 1)
#             elif m.dim() == 1:
#                 m = m.view(-1, 1)
#             elif m.dim() > 2:
#                 m = m.view(m.shape[0], -1)[:, :1]
#             data_dict["img_masks"] = m.to(torch.bool)

#         # msr3d_img_masks
#         if "msr3d_img_masks" in data_dict:
#             m = data_dict["msr3d_img_masks"]
#             if not isinstance(m, torch.Tensor):
#                 m = torch.tensor(m)
#             if m.dim() == 0:
#                 m = m.view(1, 1)
#             elif m.dim() == 1:
#                 m = m.view(-1, 1)
#             elif m.dim() > 2:
#                 m = m.view(m.shape[0], -1)[:, :1]
#             data_dict["msr3d_img_masks"] = m.to(torch.bool)
#         else:
#             data_dict["msr3d_img_masks"] = data_dict["img_masks"].clone()

#         # obj_fts
#         if "obj_fts" in data_dict:
#             if not isinstance(data_dict["obj_fts"], torch.Tensor):
#                 data_dict["obj_fts"] = torch.tensor(data_dict["obj_fts"])
#             if data_dict["obj_fts"].dim() == 3:
#                 data_dict["obj_fts"] = data_dict["obj_fts"].unsqueeze(0)
#             data_dict["obj_fts"] = data_dict["obj_fts"].float()

#         # obj_locs
#         if "obj_locs" in data_dict:
#             if not isinstance(data_dict["obj_locs"], torch.Tensor):
#                 data_dict["obj_locs"] = torch.tensor(data_dict["obj_locs"])
#             if data_dict["obj_locs"].dim() == 2:
#                 data_dict["obj_locs"] = data_dict["obj_locs"].unsqueeze(0)
#             data_dict["obj_locs"] = data_dict["obj_locs"].float()

#         # anchors
#         data_dict.setdefault("anchor_orientation", torch.zeros(4).float())
#         data_dict.setdefault("anchor_locs", torch.zeros(3).float())

#         if not isinstance(data_dict["anchor_orientation"], torch.Tensor):
#             data_dict["anchor_orientation"] = torch.tensor(data_dict["anchor_orientation"]).float()
#         if data_dict["anchor_orientation"].dim() == 1:
#             data_dict["anchor_orientation"] = data_dict["anchor_orientation"].unsqueeze(0)

#         if not isinstance(data_dict["anchor_locs"], torch.Tensor):
#             data_dict["anchor_locs"] = torch.tensor(data_dict["anchor_locs"]).float()
#         if data_dict["anchor_locs"].dim() == 1:
#             data_dict["anchor_locs"] = data_dict["anchor_locs"].unsqueeze(0)

#         data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
#         return data_dict

#     def _to_device(self, x):
#         if torch.is_tensor(x):
#             return x.to(self.device)
#         if isinstance(x, dict):
#             return {k: self._to_device(v) for k, v in x.items()}
#         if isinstance(x, (list, tuple)):
#             return type(x)(self._to_device(v) for v in x)
#         return x

#     def _compose_sample(self, base_sample, question: str, situation: str, images=None):
#         images = images or []

#         prompt = MSR3DBase.get_text_prompts(instruction=question, situation=situation)
#         prompt, _ = MSR3DBase.parse_place_holder(prompt)

#         has_imgs = isinstance(images, list) and len(images) > 0
#         img_fts = torch.stack(images) if has_imgs else torch.zeros(3, 224, 224)
#         img_masks = torch.BoolTensor([1] * len(images)) if has_imgs else torch.BoolTensor([0])
#         img_masks_2d = torch.ones(1, 1, dtype=torch.bool) if has_imgs else torch.zeros(1, 1, dtype=torch.bool)

#         data_dict = {
#             "obj_fts": base_sample["obj_fts"],
#             "obj_locs": base_sample["obj_locs"],
#             "anchor_locs": base_sample["anchor_locs"],
#             "anchor_orientation": base_sample["anchor_orientation"],
#             "scan_id": base_sample["scan_id"],
#             "img_fts": img_fts,
#             "img_masks": img_masks,
#             "msr3d_prompt": prompt,
#             "msr3d_imgs": images if has_imgs else [],
#             "msr3d_img_masks": img_masks_2d,
#             "text_output": "",
#             "answer_list": "",
#             "source": "gradio_chat",
#             "prompt_before_obj": "",
#             "prompt_middle_1": "",
#             "prompt_middle_2": "",
#             "prompt_after_obj": "",
#             "index": -1,
#             "type": "custom",
#         }

#         # This line in your original code assumes cfg.data.msqa_scannet.args.max_obj_len exists.
#         max_obj_len = self.cfg.data.msqa_scannet.args.max_obj_len
#         data_dict["obj_masks"] = (torch.arange(max_obj_len) < len(data_dict["obj_locs"])).unsqueeze(0)

#         data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
#         return data_dict

#     def answer(self, scene_id: str, question: str, situation: str) -> str:
#         idx = self._index_for_scene(scene_id)
#         base_sample = self.dataset[idx]
#         data_dict = self._compose_sample(base_sample, question=question, situation=situation, images=[])
#         data_dict = self._ensure_batched(data_dict, bs=1)
#         data_dict = self._to_device(data_dict)

#         with torch.no_grad():
#             out = self.model.generate(data_dict)

#         text = self.model.llm_tokenizer.batch_decode(out["output_tokens"], skip_special_tokens=True)
#         return text[0] if text else "No answer generated."
import os
import sys
import copy
import torch
import numpy as np
from omegaconf import OmegaConf

# keep your sys.path tweak
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.msr3d.msr3d import MSR3D
from data.datasets.msr3d import MSR3DBase, MSQAScanNet, MSQA3RScan, MSQAARkitScenes
from data.data_utils import pad_tensors


class MSR3DInteractiveService:
    """
    Load MSR3D once and answer repeatedly for the selected QA sample.
    """

    DATASET_REGISTRY = {
        "scannet": MSQAScanNet,
        "rscan": MSQA3RScan,
        "arkit": MSQAARkitScenes,
    }

    def __init__(self, experiment_path: str, split: str = "test", device: str | None = None):
        self.experiment_path = experiment_path
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg_path = os.path.join(experiment_path, "config.yaml")
        if not os.path.exists(cfg_path):
            fallback_cfg_candidates = [
                "msr3d/MSR3D_3DATASETS_FINAL_RESUME/config.yaml",
                "msr3d/MSR3D_BLIPT_PTv3_VIC_LORA_2/config.yaml",
                "MSR3D_3DATASETS_FINAL_RESUME/config.yaml",
                "MSR3D_BLIPT_PTv3_VIC_LORA_2/config.yaml",
            ]
            for cand in fallback_cfg_candidates:
                if os.path.exists(cand):
                    cfg_path = cand
                    break
        self.cfg = OmegaConf.load(cfg_path)

        self.datasets = {}
        self.current_dataset_name = "scannet"
        self.current_split = split
        self.dataset = self._get_dataset(self.current_dataset_name, split)

        self.model = MSR3D(self.cfg).to(self.device)
        ckpt_path = self._resolve_checkpoint_path(experiment_path)

        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("model", state)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def _resolve_checkpoint_path(self, experiment_path: str) -> str:
        candidates = [
            os.path.join(experiment_path, "best.pth", "pytorch_model.bin"),
            os.path.join(experiment_path, "best.pth/pytorch_model.bin"),
        ]

        exp_dir = self.cfg.get("exp_dir", None)
        if exp_dir:
            exp_dir = str(exp_dir)
            candidates.extend([
                os.path.join(exp_dir, "best.pth", "pytorch_model.bin"),
                os.path.join(exp_dir, "best.pth/pytorch_model.bin"),
            ])

        pretrain_ckpt = self.cfg.get("pretrain_ckpt_path", "")
        if pretrain_ckpt:
            pretrain_ckpt = str(pretrain_ckpt)
            candidates.extend([
                os.path.join(pretrain_ckpt, "pytorch_model.bin"),
                os.path.join(pretrain_ckpt, "best.pth", "pytorch_model.bin"),
            ])

        for cand in candidates:
            if cand and os.path.exists(cand):
                return cand

        raise FileNotFoundError(
            "Could not find a model checkpoint. Tried: " + ", ".join(candidates)
        )

    def _get_dataset(self, dataset_name: str, split: str):
        dataset_name = (dataset_name or "scannet").lower()
        if dataset_name not in self.DATASET_REGISTRY:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        split = (split or "test").lower()
        cache_key = (dataset_name, split)
        if cache_key not in self.datasets:
            self.datasets[cache_key] = self.DATASET_REGISTRY[dataset_name](self.cfg, split=split)
        return self.datasets[cache_key]

    def change_dataset(self, dataset_name: str, split: str = "test"):
        split = (split or "test").lower()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split: {split}")
        dataset_name = (dataset_name or "scannet").lower()
        if dataset_name == self.current_dataset_name and split == self.current_split:
            return
        self.dataset = self._get_dataset(dataset_name, split)
        self.current_dataset_name = dataset_name
        self.current_split = split

    def change_split(self, split: str):
        self.change_dataset(self.current_dataset_name, split=split)

    # ---------- matching helpers ----------

    def _normalize_text(self, x):
        return " ".join(str(x or "").strip().split())

    def _to_np_1d(self, x, dtype=np.float32):
        if x is None:
            return None
        if isinstance(x, dict):
            if all(k in x for k in ["x", "y", "z"]):
                return np.asarray([x["x"], x["y"], x["z"]], dtype=dtype).reshape(-1)
            if all(k in x for k in ["_x", "_y", "_z", "_w"]):
                return np.asarray([x["_x"], x["_y"], x["_z"], x["_w"]], dtype=dtype).reshape(-1)
            return None
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=dtype).reshape(-1)

    def _arr_close(self, a, b, atol=1e-5):
        aa = self._to_np_1d(a)
        bb = self._to_np_1d(b)
        if aa is None or bb is None:
            return False
        if aa.shape != bb.shape:
            return False
        return bool(np.allclose(aa, bb, atol=atol, rtol=0.0))

    def _extract_meta_anchor_locs(self, meta):
        for key in ["location", "anchor_locs", "anchor_loc", "pos", "position"]:
            if key in meta:
                arr = self._to_np_1d(meta[key])
                if arr is not None and arr.shape[0] == 3:
                    return arr
        return None

    def _extract_meta_anchor_orientation(self, meta):
        for key in ["orientation", "anchor_orientation", "ori"]:
            if key in meta:
                arr = self._to_np_1d(meta[key])
                if arr is not None and arr.shape[0] in (3, 4):
                    return arr
        return None

    def _score_meta_match(self, meta, qa_meta):
        score = 0

        if meta.get("scan_id") == qa_meta.get("scan_id"):
            score += 100

        meta_sit = self._normalize_text(meta.get("situation", ""))
        qa_sit = self._normalize_text(qa_meta.get("situation", ""))
        if qa_sit and meta_sit == qa_sit:
            score += 50

        meta_loc = self._extract_meta_anchor_locs(meta)
        qa_loc = self._to_np_1d(qa_meta.get("location", None))
        if qa_loc is not None and meta_loc is not None and self._arr_close(meta_loc, qa_loc, atol=1e-4):
            score += 20

        meta_ori = self._extract_meta_anchor_orientation(meta)
        qa_ori = self._to_np_1d(qa_meta.get("orientation", None))
        if qa_ori is not None and meta_ori is not None and self._arr_close(meta_ori, qa_ori, atol=1e-4):
            score += 20

        return score

    def _index_for_qa_meta(self, qa_meta: dict) -> int:
        if not isinstance(qa_meta, dict):
            raise TypeError(f"qa_meta must be a dict, got {type(qa_meta)}")

        scene_id = qa_meta.get("scan_id")
        if not scene_id:
            raise ValueError("qa_meta missing 'scan_id'")

        best_idx = None
        best_score = -1

        for i, meta in enumerate(self.dataset.data):
            score = self._score_meta_match(meta, qa_meta)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None or best_score < 100:
            raise ValueError(
                f"Could not find matching QA sample for scan_id={scene_id} in split={self.current_split}"
            )

        return best_idx

    # ---------- batching helpers ----------

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

    def _pad_first_dim(self, tensor: torch.Tensor, length: int, pad_value=0):
        if tensor.shape[0] > length:
            raise ValueError(f"Cannot pad tensor with first dim {tensor.shape[0]} to shorter length {length}")
        return pad_tensors(tensor, lens=length, pad=pad_value)

    def _as_singleton_list(self, value, default=""):
        if value is None:
            value = default
        if isinstance(value, list):
            if len(value) == 0:
                return [default]
            return [value[0]]
        return [value]

    # def _ensure_batched(self, data_dict, bs=1):
    #     for k in [
    #         "msr3d_prompt",
    #         "prompt_before_obj",
    #         "prompt_middle_1",
    #         "prompt_middle_2",
    #         "prompt_after_obj",
    #         "text_output",
    #         "answer_list",
    #     ]:
    #         if k in data_dict:
    #             default = "" if k != "answer_list" else ""
    #             data_dict[k] = self._broadcast_list(data_dict[k], bs, default=default)

    #     if "img_fts" in data_dict:
    #         if not isinstance(data_dict["img_fts"], torch.Tensor):
    #             data_dict["img_fts"] = torch.tensor(data_dict["img_fts"])
    #         if data_dict["img_fts"].dim() == 3:
    #             data_dict["img_fts"] = data_dict["img_fts"].unsqueeze(0)

    #     if "img_masks" not in data_dict or not isinstance(data_dict["img_masks"], torch.Tensor):
    #         has_img = (
    #             "img_fts" in data_dict
    #             and isinstance(data_dict["img_fts"], torch.Tensor)
    #             and data_dict["img_fts"].shape[0] >= 1
    #         )
    #         val = 1 if has_img else 0
    #         data_dict["img_masks"] = torch.full((bs, 1), bool(val), dtype=torch.bool)
    #     else:
    #         m = data_dict["img_masks"]
    #         if m.dim() == 0:
    #             m = m.view(1, 1)
    #         elif m.dim() == 1:
    #             m = m.view(-1, 1)
    #         elif m.dim() > 2:
    #             m = m.view(m.shape[0], -1)[:, :1]
    #         data_dict["img_masks"] = m.to(torch.bool)

    #     if "msr3d_img_masks" in data_dict:
    #         m = data_dict["msr3d_img_masks"]
    #         if not isinstance(m, torch.Tensor):
    #             m = torch.tensor(m)
    #         if m.dim() == 0:
    #             m = m.view(1, 1)
    #         elif m.dim() == 1:
    #             m = m.view(-1, 1)
    #         elif m.dim() > 2:
    #             m = m.view(m.shape[0], -1)[:, :1]
    #         data_dict["msr3d_img_masks"] = m.to(torch.bool)
    #     else:
    #         data_dict["msr3d_img_masks"] = data_dict["img_masks"].clone()

    #     if "obj_fts" in data_dict:
    #         if not isinstance(data_dict["obj_fts"], torch.Tensor):
    #             data_dict["obj_fts"] = torch.tensor(data_dict["obj_fts"])
    #         if data_dict["obj_fts"].dim() == 3:
    #             data_dict["obj_fts"] = data_dict["obj_fts"].unsqueeze(0)
    #         data_dict["obj_fts"] = data_dict["obj_fts"].float()

    #     if "obj_locs" in data_dict:
    #         if not isinstance(data_dict["obj_locs"], torch.Tensor):
    #             data_dict["obj_locs"] = torch.tensor(data_dict["obj_locs"])
    #         if data_dict["obj_locs"].dim() == 2:
    #             data_dict["obj_locs"] = data_dict["obj_locs"].unsqueeze(0)
    #         data_dict["obj_locs"] = data_dict["obj_locs"].float()

    #     if "obj_boxes" in data_dict:
    #         if not isinstance(data_dict["obj_boxes"], torch.Tensor):
    #             data_dict["obj_boxes"] = torch.tensor(data_dict["obj_boxes"])
    #         if data_dict["obj_boxes"].dim() == 2:
    #             data_dict["obj_boxes"] = data_dict["obj_boxes"].unsqueeze(0)
    #         data_dict["obj_boxes"] = data_dict["obj_boxes"].float()

    #     if "scene_fts" in data_dict:
    #         if not isinstance(data_dict["scene_fts"], torch.Tensor):
    #             data_dict["scene_fts"] = torch.tensor(data_dict["scene_fts"])
    #         if data_dict["scene_fts"].dim() == 2:
    #             data_dict["scene_fts"] = data_dict["scene_fts"].unsqueeze(0)
    #         data_dict["scene_fts"] = data_dict["scene_fts"].float()

    #     if "scene_pcds" not in data_dict and "scene_fts" in data_dict:
    #         data_dict["scene_pcds"] = data_dict["scene_fts"]

    #     if "scene_pcds" in data_dict:
    #         if not isinstance(data_dict["scene_pcds"], torch.Tensor):
    #             data_dict["scene_pcds"] = torch.tensor(data_dict["scene_pcds"])
    #         if data_dict["scene_pcds"].dim() == 2:
    #             data_dict["scene_pcds"] = data_dict["scene_pcds"].unsqueeze(0)
    #         data_dict["scene_pcds"] = data_dict["scene_pcds"].float()

    #     if "instance_ids" in data_dict:
    #         if not isinstance(data_dict["instance_ids"], torch.Tensor):
    #             data_dict["instance_ids"] = torch.tensor(data_dict["instance_ids"])
    #         if data_dict["instance_ids"].dim() == 1:
    #             data_dict["instance_ids"] = data_dict["instance_ids"].unsqueeze(0)
    #         data_dict["instance_ids"] = data_dict["instance_ids"].long()

    #     if "segments" in data_dict:
    #         if not isinstance(data_dict["segments"], torch.Tensor):
    #             data_dict["segments"] = torch.tensor(data_dict["segments"])
    #         if data_dict["segments"].dim() == 1:
    #             data_dict["segments"] = data_dict["segments"].unsqueeze(0)
    #         data_dict["segments"] = data_dict["segments"].long()

    #     data_dict.setdefault("anchor_orientation", torch.zeros(4).float())
    #     data_dict.setdefault("anchor_locs", torch.zeros(3).float())

    #     if not isinstance(data_dict["anchor_orientation"], torch.Tensor):
    #         data_dict["anchor_orientation"] = torch.tensor(data_dict["anchor_orientation"]).float()
    #     if data_dict["anchor_orientation"].dim() == 1:
    #         data_dict["anchor_orientation"] = data_dict["anchor_orientation"].unsqueeze(0)

    #     if not isinstance(data_dict["anchor_locs"], torch.Tensor):
    #         data_dict["anchor_locs"] = torch.tensor(data_dict["anchor_locs"]).float()
    #     if data_dict["anchor_locs"].dim() == 1:
    #         data_dict["anchor_locs"] = data_dict["anchor_locs"].unsqueeze(0)

    #     if "obj_masks" in data_dict:
    #         m = data_dict["obj_masks"]
    #         if not isinstance(m, torch.Tensor):
    #             m = torch.tensor(m)
    #         if m.dim() == 1:
    #             m = m.unsqueeze(0)
    #         data_dict["obj_masks"] = m.to(torch.bool)
    #     else:
    #         cur_obj_len = 0
    #         if "obj_locs" in data_dict and isinstance(data_dict["obj_locs"], torch.Tensor):
    #             cur_obj_len = int(data_dict["obj_locs"].shape[1])
    #         elif "obj_fts" in data_dict and isinstance(data_dict["obj_fts"], torch.Tensor):
    #             cur_obj_len = int(data_dict["obj_fts"].shape[1])

    #         max_obj_len = int(getattr(self.cfg.data.msqa_scannet.args, "max_obj_len", cur_obj_len))
    #         max_obj_len = max(max_obj_len, cur_obj_len)
    #         data_dict["obj_masks"] = (torch.arange(max_obj_len) < cur_obj_len).unsqueeze(0)

    #     data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
    #     return data_dict
    def _prepare_single_sample(self, data_dict):
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
                data_dict[k] = self._as_singleton_list(data_dict[k], default=default)

        for k, default in [
            ("scan_id", ""),
            ("source", ""),
            ("type", ""),
            ("index", -1),
        ]:
            if k in data_dict:
                data_dict[k] = self._as_singleton_list(data_dict[k], default=default)

        # Single-scene interactive inference still needs a batch dimension for MSR3D.generate().
        if "img_fts" in data_dict:
            if not isinstance(data_dict["img_fts"], torch.Tensor):
                data_dict["img_fts"] = torch.tensor(data_dict["img_fts"])
            if data_dict["img_fts"].dim() == 3:
                data_dict["img_fts"] = data_dict["img_fts"].unsqueeze(0)
            data_dict["img_fts"] = data_dict["img_fts"].float()

        max_img_num = int(getattr(self.cfg.data.msqa_scannet.args, "msr3d_max_img_num", 10))
        if "msr3d_imgs" in data_dict:
            imgs = data_dict["msr3d_imgs"]
            if isinstance(imgs, list):
                num_imgs = len(imgs)
                if num_imgs == 0:
                    data_dict["msr3d_imgs"] = torch.zeros((max_img_num, 3, 224, 224), dtype=torch.float32)
                else:
                    stacked = torch.stack(imgs, dim=0).float()
                    data_dict["msr3d_imgs"] = self._pad_first_dim(stacked, max_img_num, pad_value=0.0)
                data_dict["msr3d_img_masks"] = (torch.arange(max_img_num) < num_imgs).unsqueeze(0)
            elif isinstance(imgs, torch.Tensor):
                if imgs.dim() == 3:
                    imgs = imgs.unsqueeze(0)
                num_imgs = int(imgs.shape[0])
                data_dict["msr3d_imgs"] = self._pad_first_dim(imgs.float(), max_img_num, pad_value=0.0)
                if "msr3d_img_masks" not in data_dict:
                    data_dict["msr3d_img_masks"] = (torch.arange(max_img_num) < num_imgs).unsqueeze(0)

        if "img_masks" not in data_dict or not isinstance(data_dict["img_masks"], torch.Tensor):
            has_img = (
                "img_fts" in data_dict
                and isinstance(data_dict["img_fts"], torch.Tensor)
                and data_dict["img_fts"].shape[0] >= 1
            )
            val = 1 if has_img else 0
            data_dict["img_masks"] = torch.full((1, 1), bool(val), dtype=torch.bool)
        else:
            m = data_dict["img_masks"]
            if m.dim() == 0:
                m = m.view(1, 1)
            elif m.dim() == 1:
                m = m.view(-1, 1)
            elif m.dim() > 2:
                m = m.view(m.shape[0], -1)[:, :1]
            data_dict["img_masks"] = m.to(torch.bool)

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

        # ---------- object tensors ----------
        max_obj_len = int(getattr(self.cfg.data.msqa_scannet.args, "max_obj_len", 0))
        cur_obj_len = 0

        if "obj_fts" in data_dict:
            if not isinstance(data_dict["obj_fts"], torch.Tensor):
                data_dict["obj_fts"] = torch.tensor(data_dict["obj_fts"])
            obj_fts = data_dict["obj_fts"].float()
            if obj_fts.dim() == 4:
                if obj_fts.shape[0] != 1:
                    raise ValueError(f"Interactive path expects one scene, got obj_fts batch size {obj_fts.shape[0]}")
                obj_fts = obj_fts[0]
            if obj_fts.dim() != 3:
                raise ValueError(f"obj_fts must have shape (num_obj, num_pts, feat_dim), got {tuple(obj_fts.shape)}")
            cur_obj_len = int(obj_fts.shape[0])
            if max_obj_len <= 0:
                max_obj_len = cur_obj_len
            obj_fts = self._pad_first_dim(obj_fts, max_obj_len, pad_value=1.0)
            data_dict["obj_fts"] = obj_fts.unsqueeze(0)

        if "obj_locs" in data_dict:
            if not isinstance(data_dict["obj_locs"], torch.Tensor):
                data_dict["obj_locs"] = torch.tensor(data_dict["obj_locs"])
            obj_locs = data_dict["obj_locs"].float()
            if obj_locs.dim() == 3:
                if obj_locs.shape[0] != 1:
                    raise ValueError(f"Interactive path expects one scene, got obj_locs batch size {obj_locs.shape[0]}")
                obj_locs = obj_locs[0]
            if obj_locs.dim() != 2:
                raise ValueError(f"obj_locs must have shape (num_obj, 6), got {tuple(obj_locs.shape)}")
            cur_obj_len = max(cur_obj_len, int(obj_locs.shape[0]))
            if max_obj_len <= 0:
                max_obj_len = int(obj_locs.shape[0])
            obj_locs = self._pad_first_dim(obj_locs, max_obj_len, pad_value=0.0)
            data_dict["obj_locs"] = obj_locs.unsqueeze(0)

        if "obj_boxes" in data_dict:
            if not isinstance(data_dict["obj_boxes"], torch.Tensor):
                data_dict["obj_boxes"] = torch.tensor(data_dict["obj_boxes"])
            if data_dict["obj_boxes"].dim() == 2:
                data_dict["obj_boxes"] = data_dict["obj_boxes"].unsqueeze(0)
            data_dict["obj_boxes"] = data_dict["obj_boxes"].float()

        # ---------- scene-level tensors ----------
        if "scene_fts" in data_dict:
            if not isinstance(data_dict["scene_fts"], torch.Tensor):
                data_dict["scene_fts"] = torch.tensor(data_dict["scene_fts"])
            scene_fts = data_dict["scene_fts"].float()
            if scene_fts.dim() == 3:
                if scene_fts.shape[0] != 1:
                    raise ValueError(f"Interactive path expects one scene, got scene_fts batch size {scene_fts.shape[0]}")
                scene_fts = scene_fts[0]
            if scene_fts.dim() != 2:
                raise ValueError(f"scene_fts must have shape (num_scene_points, feat_dim), got {tuple(scene_fts.shape)}")
            data_dict["scene_fts"] = scene_fts

        if "scene_pcds" not in data_dict and "scene_fts" in data_dict:
            data_dict["scene_pcds"] = data_dict["scene_fts"]

        if "scene_pcds" in data_dict:
            if not isinstance(data_dict["scene_pcds"], torch.Tensor):
                data_dict["scene_pcds"] = torch.tensor(data_dict["scene_pcds"])
            scene_pcds = data_dict["scene_pcds"].float()
            if scene_pcds.dim() == 3:
                if scene_pcds.shape[0] != 1:
                    raise ValueError(f"Interactive path expects one scene, got scene_pcds batch size {scene_pcds.shape[0]}")
                scene_pcds = scene_pcds[0]
            if scene_pcds.dim() != 2:
                raise ValueError(f"scene_pcds must have shape (num_scene_points, feat_dim), got {tuple(scene_pcds.shape)}")
            data_dict["scene_pcds"] = scene_pcds

        if "instance_ids" in data_dict:
            if not isinstance(data_dict["instance_ids"], torch.Tensor):
                data_dict["instance_ids"] = torch.tensor(data_dict["instance_ids"])
            instance_ids = data_dict["instance_ids"].long()
            if instance_ids.dim() == 2:
                if instance_ids.shape[0] != 1:
                    raise ValueError(
                        f"Interactive path expects one scene, got instance_ids batch size {instance_ids.shape[0]}"
                    )
                instance_ids = instance_ids[0]
            if instance_ids.dim() != 1:
                raise ValueError(f"instance_ids must have shape (num_scene_points,), got {tuple(instance_ids.shape)}")
            data_dict["instance_ids"] = instance_ids

        if "segments" in data_dict:
            if not isinstance(data_dict["segments"], torch.Tensor):
                data_dict["segments"] = torch.tensor(data_dict["segments"])
            segments = data_dict["segments"].long()
            if segments.dim() == 2:
                if segments.shape[0] != 1:
                    raise ValueError(f"Interactive path expects one scene, got segments batch size {segments.shape[0]}")
                segments = segments[0]
            if segments.dim() != 1:
                raise ValueError(f"segments must have shape (num_scene_points,), got {tuple(segments.shape)}")
            data_dict["segments"] = segments

        if "selected_obj_ids" in data_dict:
            if not isinstance(data_dict["selected_obj_ids"], torch.Tensor):
                data_dict["selected_obj_ids"] = torch.tensor(data_dict["selected_obj_ids"])
            selected_obj_ids = data_dict["selected_obj_ids"].long()
            if selected_obj_ids.dim() == 2:
                if selected_obj_ids.shape[0] != 1:
                    raise ValueError(
                        f"Interactive path expects one scene, got selected_obj_ids batch size {selected_obj_ids.shape[0]}"
                    )
                selected_obj_ids = selected_obj_ids[0]
            if selected_obj_ids.dim() != 1:
                raise ValueError(
                    f"selected_obj_ids must have shape (num_obj,), got {tuple(selected_obj_ids.shape)}"
                )
            if max_obj_len <= 0:
                max_obj_len = int(selected_obj_ids.shape[0])
            selected_obj_ids = self._pad_first_dim(selected_obj_ids, max_obj_len, pad_value=-1)
            data_dict["selected_obj_ids"] = selected_obj_ids.unsqueeze(0)

        if "scene_fts" in data_dict and "scene_offset" not in data_dict:
            data_dict["scene_offset"] = torch.tensor([data_dict["scene_fts"].shape[0]], dtype=torch.int32)
        elif "scene_offset" in data_dict:
            if not isinstance(data_dict["scene_offset"], torch.Tensor):
                data_dict["scene_offset"] = torch.tensor(data_dict["scene_offset"])
            scene_offset = data_dict["scene_offset"].long().view(-1)
            if scene_offset.numel() != 1 and scene_offset.numel() > 1:
                scene_offset = torch.tensor([int(scene_offset[-1].item())], dtype=scene_offset.dtype)
            data_dict["scene_offset"] = scene_offset

        # ---------- anchors ----------
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

        # ---------- object masks ----------
        if "obj_masks" in data_dict:
            m = data_dict["obj_masks"]
            if not isinstance(m, torch.Tensor):
                m = torch.tensor(m)
            if m.dim() == 1:
                m = m.unsqueeze(0)
            if max_obj_len > 0 and m.shape[1] < max_obj_len:
                pad = torch.zeros((m.shape[0], max_obj_len - m.shape[1]), dtype=m.dtype, device=m.device)
                m = torch.cat([m, pad], dim=1)
            data_dict["obj_masks"] = m.to(torch.bool)
        else:
            if cur_obj_len == 0 and "obj_locs" in data_dict and isinstance(data_dict["obj_locs"], torch.Tensor):
                cur_obj_len = int((data_dict["obj_locs"][0].abs().sum(dim=-1) > 0).sum().item())
            elif cur_obj_len == 0 and "selected_obj_ids" in data_dict and isinstance(data_dict["selected_obj_ids"], torch.Tensor):
                cur_obj_len = int((data_dict["selected_obj_ids"][0] >= 0).sum().item())
            elif cur_obj_len == 0 and "obj_fts" in data_dict and isinstance(data_dict["obj_fts"], torch.Tensor):
                cur_obj_len = int(data_dict["obj_fts"].shape[1])

            max_obj_len = max(max_obj_len, cur_obj_len)
            data_dict["obj_masks"] = (torch.arange(max_obj_len) < cur_obj_len).unsqueeze(0)

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

        # Preserve the full tested __getitem__ output
        data_dict = copy.deepcopy(base_sample)

        # Override only interactive fields
        data_dict["scan_id"] = base_sample.get("scan_id", "")
        data_dict["img_fts"] = img_fts
        data_dict["img_masks"] = img_masks
        data_dict["msr3d_prompt"] = prompt
        data_dict["msr3d_imgs"] = images if has_imgs else []
        data_dict["msr3d_img_masks"] = img_masks_2d
        data_dict["text_output"] = ""
        data_dict["answer_list"] = ""
        data_dict["source"] = "gradio_chat"
        data_dict["type"] = "custom"
        data_dict["index"] = data_dict.get("index", -1)

        data_dict["prompt_before_obj"] = data_dict.get("prompt_before_obj", "")
        data_dict["prompt_middle_1"] = data_dict.get("prompt_middle_1", "")
        data_dict["prompt_middle_2"] = data_dict.get("prompt_middle_2", "")
        data_dict["prompt_after_obj"] = data_dict.get("prompt_after_obj", "")

        # Keep useful debug fields
        data_dict["interactive_question"] = question
        data_dict["interactive_situation"] = situation

        # Alias if old downstream code expects it
        if "scene_pcds" not in data_dict and "scene_fts" in data_dict:
            data_dict["scene_pcds"] = data_dict["scene_fts"]

        data_dict = MSR3DBase.check_output_and_fill_dummy(data_dict)
        return data_dict

    def answer(self, qa_meta: dict, question: str, situation: str | None = None, images=None) -> str:
        if not isinstance(qa_meta, dict):
            raise TypeError(f"qa_meta must be dict, got {type(qa_meta)}")

        dataset_name = qa_meta.get("dataset_name", self.current_dataset_name)
        target_split = qa_meta.get("split", self.current_split)
        if dataset_name != self.current_dataset_name or target_split != self.current_split:
            self.change_dataset(dataset_name, split=target_split)

        if situation is None:
            situation = qa_meta.get("situation", "")

        idx = self._index_for_qa_meta(qa_meta)
        base_sample = self.dataset[idx]

        data_dict = self._compose_sample(
            base_sample=base_sample,
            question=question,
            situation=situation,
            images=images or [],
        )
        data_dict = self._prepare_single_sample(data_dict)
        data_dict = self._to_device(data_dict)

        with torch.no_grad():
            out = self.model.generate(data_dict)

        text = self.model.llm_tokenizer.batch_decode(out["output_tokens"], skip_special_tokens=True)
        return text[0] if text else "No answer generated."

    def make_chat_callback(self, qa_meta: dict, situation: str | None = None, images=None):
        return lambda user_text: self.answer(
            qa_meta=qa_meta,
            question=user_text,
            situation=situation,
            images=images,
        )
