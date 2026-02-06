# #!/usr/bin/env python3
# """
# MSQA Multi-Dataset Scene Viewer (Gradio + Plotly) — ScanNet + ARKitScenes + RScan (folder-based)

# What you asked for (implemented):
# - For the RScan dataset: the scene dropdown shows the folder names (scan_id).
# - When rendering an RScan scene, it loads: <RSCAN_PCD_ROOT>/<scan_id>/pcds.pth

# Other datasets:
# - ScanNet / ARKit: still load from <PCD_ROOT>/<scan_id>.pth

# UI additions retained:
# 1) Visualization details hidden by default (Accordion) + Show/Hide toggle button.
# 2) Right-side chat stub.
# """

# import os
# import json
# import torch
# import numpy as np
# import gradio as gr
# import plotly.graph_objects as go
# from scipy.spatial.transform import Rotation as R
# from collections import defaultdict
# import pandas as pd

# # ======================== Config ========================

# # SCANNET_ROOT_DIR = "/mnt/d/Thesis/data/text_annotations/msqa/scannet"
# # ARKIT_ROOT_DIR   = "/mnt/d/Thesis/data/text_annotations/msqa/arkitscenes"
# # RSCAN_ROOT_DIR   = "/mnt/d/Thesis/data/text_annotations/msqa/rscan"
# SCANNET_ROOT_DIR = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/text_annotations/msqa/scannet"
# ARKIT_ROOT_DIR   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/text_annotations/msqa/arkitscenes"
# RSCAN_ROOT_DIR   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/text_annotations/msqa/rscan"


# SCANNET_JSON_PATHS = {
#     "train": os.path.join(SCANNET_ROOT_DIR, "msqa_scannet_train.json"),
#     "val":   os.path.join(SCANNET_ROOT_DIR, "msqa_scannet_val.json"),
#     "test":  os.path.join(SCANNET_ROOT_DIR, "msqa_scannet_test.json"),
# }
# ARKIT_JSON_PATHS = {
#     "train": os.path.join(ARKIT_ROOT_DIR, "msqa_arkitscenes_train.json"),
#     "val":   os.path.join(ARKIT_ROOT_DIR, "msqa_arkitscenes_val.json"),
#     "test":  os.path.join(ARKIT_ROOT_DIR, "msqa_arkitscenes_test.json"),
# }
# RSCAN_JSON_PATHS = {
#     "train": os.path.join(RSCAN_ROOT_DIR, "msqa_rscan_train.json"),
#     "val":   os.path.join(RSCAN_ROOT_DIR, "msqa_rscan_val.json"),
#     "test":  os.path.join(RSCAN_ROOT_DIR, "msqa_rscan_test.json"),
# }

# # SCANNET_PCD_ROOT = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment"
# # ARKIT_PCD_ROOT   = "/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd-align/pcd-align"
# SCANNET_PCD_ROOT = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment"
# ARKIT_PCD_ROOT   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd-align/"

# # IMPORTANT: RScan is folder-based, and we will load pcds.pth inside each folder
# # RSCAN_PCD_ROOT   = "/mnt/d/Thesis/data/MSR3D_v2_pcds/rscan_base/3RScan-ours-align/3RScan-ours-align"
# RSCAN_PCD_ROOT   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/rscan_base/3RScan-ours-align/"
# RSCAN_PCD_FILE   = "pcds.pth"  # you explicitly requested this file

# ONLY_SHOW_SCANS_WITH_PTH = True

# DATASET_SPECS = {
#     "scannet": {"json_paths": SCANNET_JSON_PATHS, "pcd_root": SCANNET_PCD_ROOT},
#     "arkit":   {"json_paths": ARKIT_JSON_PATHS,   "pcd_root": ARKIT_PCD_ROOT},
#     "rscan":   {"json_paths": RSCAN_JSON_PATHS,   "pcd_root": RSCAN_PCD_ROOT},
# }


# import threading
# from msr3d.tools.interactive_service import MSR3DInteractiveService

# MSR3D_EXPERIMENT_PATH = "MSR3D_BLIPT_PTv3_VIC_LORA_2"

# # One shared model instance
# MSR3D_SERVICE = None
# MSR3D_LOCK = threading.Lock()

# def get_msr3d_service():
#     global MSR3D_SERVICE
#     if MSR3D_SERVICE is None:
#         MSR3D_SERVICE = MSR3DInteractiveService(
#             experiment_path=MSR3D_EXPERIMENT_PATH,
#             split="test",
#         )
#     return MSR3D_SERVICE
# # ======================== Utils ========================

# def load_json(path: str):
#     with open(path, "r") as f:
#         return json.load(f)

# def ensure_np(x):
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)

# def normalize_rgb01(colors):
#     """
#     Accept colors in:
#       - [-1,1] float
#       - [0,1] float
#       - [0,255] float/uint8
#     Return float RGB in [0,1].
#     """
#     c = ensure_np(colors).astype(np.float32).reshape(-1, 3)
#     cmin, cmax = float(np.nanmin(c)), float(np.nanmax(c))

#     if cmin >= -1.01 and cmax <= 1.01:
#         # either [-1,1] or [0,1]
#         if cmin < 0.0:
#             return np.clip((c + 1.0) * 0.5, 0.0, 1.0)
#         return np.clip(c, 0.0, 1.0)

#     # assume [0,255]
#     return np.clip(c / 255.0, 0.0, 1.0)

# def hash_colors_for_labels(labels, seed=0):
#     labels = np.asarray(labels).reshape(-1)
#     rng = np.random.default_rng(seed)
#     uniq = np.unique(labels)
#     table = {}
#     for u in uniq:
#         col = rng.random(3) * 0.8 + 0.2
#         table[int(u)] = col.astype(np.float32)
#     return np.array([table[int(x)] for x in labels], dtype=np.float32)

# # -------- Orientation handling --------

# def get_view_vector_from_orientation(orientation):
#     o = ensure_np(orientation).astype(np.float32).reshape(-1)

#     if o.size == 4:
#         yaw = R.from_quat(o).as_euler("xyz", degrees=False)[-1]
#         d = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
#         return d / (np.linalg.norm(d) + 1e-12)

#     if o.size == 3:
#         n = float(np.linalg.norm(o))
#         if 0.5 <= n <= 1.5:
#             return (o / (n + 1e-12)).astype(np.float32)
#         yaw = float(o[-1])
#         d = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
#         return d / (np.linalg.norm(d) + 1e-12)

#     return np.array([1.0, 0.0, 0.0], dtype=np.float32)

# # ======================== Plotly overlay geometry ========================

# def aabb_edges(minb, maxb):
#     minb = np.asarray(minb, dtype=np.float32)
#     maxb = np.asarray(maxb, dtype=np.float32)

#     x0, y0, z0 = minb
#     x1, y1, z1 = maxb

#     corners = np.array([
#         [x0, y0, z0],
#         [x1, y0, z0],
#         [x1, y1, z0],
#         [x0, y1, z0],
#         [x0, y0, z1],
#         [x1, y0, z1],
#         [x1, y1, z1],
#         [x0, y1, z1],
#     ], dtype=np.float32)

#     edges = [
#         (0, 1), (1, 2), (2, 3), (3, 0),
#         (4, 5), (5, 6), (6, 7), (7, 4),
#         (0, 4), (1, 5), (2, 6), (3, 7),
#     ]
#     return [(corners[i], corners[j]) for i, j in edges]

# def build_instance_bboxes_as_traces(xyz, instance_labels, max_boxes=200, seed=123):
#     if instance_labels is None:
#         return []

#     xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
#     inst = np.asarray(instance_labels).reshape(-1)
#     uniq = np.unique(inst)

#     uniq_cols = hash_colors_for_labels(uniq, seed=seed)
#     col_tab = {int(u): uniq_cols[i] for i, u in enumerate(uniq)}

#     traces = []
#     if len(uniq) > max_boxes:
#         uniq = uniq[:max_boxes]

#     for u in uniq:
#         mask = (inst == u)
#         if not np.any(mask):
#             continue
#         pts = xyz[mask]
#         if pts.shape[0] < 2:
#             continue

#         minb = pts.min(0)
#         maxb = pts.max(0)
#         segs = aabb_edges(minb, maxb)
#         col = col_tab[int(u)]

#         xs, ys, zs = [], [], []
#         for p0, p1 in segs:
#             xs += [p0[0], p1[0], None]
#             ys += [p0[1], p1[1], None]
#             zs += [p0[2], p1[2], None]

#         traces.append(go.Scatter3d(
#             x=xs, y=ys, z=zs,
#             mode="lines",
#             line=dict(width=4, color=f"rgb({int(col[0]*255)},{int(col[1]*255)},{int(col[2]*255)})"),
#             showlegend=False,
#         ))

#     return traces

# def make_world_axis_traces(origin, axis_len=1.0):
#     origin = np.asarray(origin, dtype=np.float32).reshape(3)
#     O = origin
#     X = origin + np.array([axis_len, 0, 0], dtype=np.float32)
#     Y = origin + np.array([0, axis_len, 0], dtype=np.float32)
#     Z = origin + np.array([0, 0, axis_len], dtype=np.float32)

#     def line(p0, p1, color):
#         return go.Scatter3d(
#             x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
#             mode="lines",
#             line=dict(width=8, color=color),
#             showlegend=False,
#         )

#     return [line(O, X, "red"), line(O, Y, "green"), line(O, Z, "blue")]

# def make_situation_arrow_trace(location, orientation, scale=0.8):
#     loc = np.asarray(location, dtype=np.float32).reshape(3)
#     d = get_view_vector_from_orientation(orientation)

#     loc2 = loc.copy()
#     loc2[2] += float(0.15)
#     tip = loc2 + d * float(scale)

#     return go.Scatter3d(
#         x=[loc2[0], tip[0]],
#         y=[loc2[1], tip[1]],
#         z=[loc2[2], tip[2]],
#         mode="lines+markers",
#         line=dict(width=10, color="orange"),
#         marker=dict(size=4),
#         showlegend=False,
#     )

# # ======================== Dataset path resolution ========================

# def resolve_pth_path(dataset_name: str, scan_id: str) -> str:
#     """
#     What you asked for:
#     - RScan: load <RSCAN_PCD_ROOT>/<scan_id>/pcds.pth
#     - Others: load <PCD_ROOT>/<scan_id>.pth
#     """
#     if dataset_name == "rscan":
#         return os.path.join(RSCAN_PCD_ROOT, scan_id, RSCAN_PCD_FILE)

#     root = DATASET_SPECS[dataset_name]["pcd_root"]
#     return os.path.join(root, f"{scan_id}.pth")

# def pth_exists(dataset_name: str, scan_id: str) -> bool:
#     return os.path.exists(resolve_pth_path(dataset_name, scan_id))

# # ======================== Load + merge splits (per dataset) ========================

# DATA_BY_DATASET = {}
# SCAN_TO_INDICES_BY_DATASET = {}
# AVAILABLE_SCANS_BY_DATASET = {}

# def build_scan_index(data):
#     scan_to_indices = defaultdict(list)
#     for i, qa in enumerate(data):
#         scan_to_indices[qa["scan_id"]].append(i)
#     return dict(scan_to_indices)

# for dname, spec in DATASET_SPECS.items():
#     data = []
#     for split_name, path in spec["json_paths"].items():
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Missing JSON for dataset '{dname}' split '{split_name}': {path}")
#         items = load_json(path)
#         for it in items:
#             it = dict(it)
#             it["split"] = split_name
#             data.append(it)

#     DATA_BY_DATASET[dname] = data
#     SCAN_TO_INDICES_BY_DATASET[dname] = build_scan_index(data)

#     all_scans = sorted(SCAN_TO_INDICES_BY_DATASET[dname].keys())
#     if ONLY_SHOW_SCANS_WITH_PTH:
#         avail = [sid for sid in all_scans if pth_exists(dname, sid)]
#     else:
#         avail = all_scans

#     if len(avail) == 0:
#         raise RuntimeError(f"No scan_ids available for dataset '{dname}'. Check JSON paths and PCD roots.")
#     AVAILABLE_SCANS_BY_DATASET[dname] = avail

# # ======================== QA dropdown helpers ========================

# def qa_label(dataset_name: str, i: int) -> str:
#     qa = DATA_BY_DATASET[dataset_name][i]
#     sit = (qa.get("situation", "") or "").strip().replace("\n", " ")
#     if len(sit) > 90:
#         sit = sit[:87] + "..."
#     return f"{i} | {qa['scan_id']} | {qa.get('split','?')} | {sit}"

# def qa_choices_for_scan(dataset_name: str, scan_id: str, split_filter: str):
#     inds = SCAN_TO_INDICES_BY_DATASET[dataset_name].get(scan_id, [])
#     if split_filter != "all":
#         inds = [i for i in inds if DATA_BY_DATASET[dataset_name][i].get("split") == split_filter]
#     return [(qa_label(dataset_name, i), i) for i in inds]

# # ======================== Scene load + render ========================

# def load_scene(dataset_name: str, idx: int):
#     qa = DATA_BY_DATASET[dataset_name][idx]
#     scan_id = qa["scan_id"]

#     pth_path = resolve_pth_path(dataset_name, scan_id)
#     if not os.path.exists(pth_path):
#         raise FileNotFoundError(f"Missing PTH: {pth_path}")

#     pcd_data = torch.load(pth_path, weights_only=False)

#     if not isinstance(pcd_data, (tuple, list)) or len(pcd_data) < 2:
#         raise ValueError(f"Unsupported PTH format for {pth_path}: {type(pcd_data)}")

#     points = ensure_np(pcd_data[0]).astype(np.float32).reshape(-1, 3)
#     colors = ensure_np(pcd_data[1]).astype(np.float32).reshape(-1, 3)
#     rgb01 = normalize_rgb01(colors)

#     # Attempt labels (best-effort)
#     instance_labels = None
#     if len(pcd_data) >= 3:
#         cand_last = ensure_np(pcd_data[-1]).reshape(-1)
#         if cand_last.shape[0] == points.shape[0]:
#             instance_labels = cand_last.astype(np.int32)
#         else:
#             cand2 = ensure_np(pcd_data[2]).reshape(-1)
#             if cand2.shape[0] == points.shape[0]:
#                 instance_labels = cand2.astype(np.int32)

#     location = qa.get("location", None)
#     orientation = qa.get("orientation", None)

#     if isinstance(location, dict):
#         location = [location.get("x", 0.0), location.get("y", 0.0), location.get("z", 0.0)]

#     if isinstance(orientation, dict):
#         if all(k in orientation for k in ["_x", "_y", "_z", "_w"]):
#             orientation = [orientation["_x"], orientation["_y"], orientation["_z"], orientation["_w"]]
#         elif all(k in orientation for k in ["x", "y", "z"]):
#             orientation = [orientation["x"], orientation["y"], orientation["z"]]

#     return {
#         "scan_id": scan_id,
#         "split": qa.get("split", "unknown"),
#         "situation": qa.get("situation", ""),
#         "points": points,
#         "rgb01": rgb01,
#         "instance_labels": instance_labels,
#         "segment20": None,
#         "segment200": None,
#         "location": location,
#         "orientation": orientation,
#     }

# def build_plotly_figure(scene, color_mode: str, point_size: float,
#                         show_boxes: bool, show_axis: bool, show_arrow: bool,
#                         axis_len: float, max_points: int, max_boxes: int):

#     xyz = scene["points"].reshape(-1, 3)
#     N = xyz.shape[0]

#     if max_points is not None and max_points > 0 and N > max_points:
#         idx = np.random.default_rng(0).choice(N, size=int(max_points), replace=False)
#         xyz_vis = xyz[idx]
#         rgb_vis = scene["rgb01"][idx]
#         inst_vis = scene["instance_labels"][idx] if scene["instance_labels"] is not None else None
#     else:
#         xyz_vis = xyz
#         rgb_vis = scene["rgb01"]
#         inst_vis = scene["instance_labels"]

#     if color_mode == "RGB":
#         cols = rgb_vis
#     elif color_mode == "Instance":
#         cols = rgb_vis if inst_vis is None else hash_colors_for_labels(inst_vis, seed=123)
#     else:
#         cols = rgb_vis

#     cols255 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
#     color_str = [f"rgb({r},{g},{b})" for r, g, b in cols255]

#     fig = go.Figure()
#     fig.add_trace(go.Scatter3d(
#         x=xyz_vis[:, 0], y=xyz_vis[:, 1], z=xyz_vis[:, 2],
#         mode="markers",
#         marker=dict(size=float(point_size), color=color_str, opacity=1.0),
#         showlegend=False,
#     ))

#     if show_boxes and scene["instance_labels"] is not None:
#         for t in build_instance_bboxes_as_traces(scene["points"], scene["instance_labels"], max_boxes=int(max_boxes)):
#             fig.add_trace(t)

#     if show_axis:
#         center = xyz_vis.mean(axis=0)
#         for t in make_world_axis_traces(center, axis_len=float(axis_len)):
#             fig.add_trace(t)

#     if show_arrow and scene["location"] is not None and scene["orientation"] is not None:
#         try:
#             fig.add_trace(make_situation_arrow_trace(scene["location"], scene["orientation"], scale=float(axis_len)))
#         except Exception as e:
#             print(f"[warn] could not render situation arrow: {e}")

#     title = f"{scene['scan_id']} | split: {scene.get('split','?')}"
#     if scene.get("situation"):
#         title += f" | {scene['situation']}"

#     fig.update_layout(
#         title=title,
#         margin=dict(l=0, r=0, b=0, t=40),
#         scene=dict(
#             aspectmode="data",
#             xaxis=dict(visible=False),
#             yaxis=dict(visible=False),
#             zaxis=dict(visible=False),
#             bgcolor="black",
#         ),
#         paper_bgcolor="black",
#         font=dict(color="white"),
#     )
    
#     return fig

# # ======================== Gradio callbacks ========================
# def scans_for_split(dataset_name: str, split_filter: str):
#     data = DATA_BY_DATASET[dataset_name]
#     scan_to_inds = SCAN_TO_INDICES_BY_DATASET[dataset_name]

#     if split_filter == "all":
#         scans = list(scan_to_inds.keys())
#     else:
#         scans = []
#         for sid, inds in scan_to_inds.items():
#             if any(data[i].get("split") == split_filter for i in inds):
#                 scans.append(sid)

#     scans = sorted(scans)  # nice ascending order

#     if ONLY_SHOW_SCANS_WITH_PTH:
#         scans = [sid for sid in scans if pth_exists(dataset_name, sid)]

#     return scans
# def on_split_change(dataset_name: str, split_filter: str):
#     scans = scans_for_split(dataset_name, split_filter)
#     scan_val = scans[0] if scans else None

#     qa_choices = qa_choices_for_scan(dataset_name, scan_val, split_filter) if scan_val else []
#     qa_val = qa_choices[0][1] if qa_choices else None

#     return (
#         gr.update(choices=scans, value=scan_val),      # scan_id_dd
#         gr.update(choices=qa_choices, value=qa_val),   # qa_dd
#     )


# def on_dataset_change(dataset_name: str):
#     scans = AVAILABLE_SCANS_BY_DATASET[dataset_name]
#     scan_val = scans[0] if scans else None
#     split_val = "all"

#     qa_choices = qa_choices_for_scan(dataset_name, scan_val, split_val) if scan_val else []
#     qa_val = qa_choices[0][1] if qa_choices else None

#     return (
#         gr.update(choices=scans, value=scan_val),     # scan_id_dd
#         gr.update(value=split_val),                   # split_filter
#         gr.update(choices=qa_choices, value=qa_val),  # qa_dd
#     )

# def on_scan_or_split_change(dataset_name: str, scan_id: str, split_filter: str):
#     choices = qa_choices_for_scan(dataset_name, scan_id, split_filter)
#     default_val = choices[0][1] if choices else None
#     return gr.update(choices=choices, value=default_val)

# def render(dataset_name, global_idx, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_points, max_boxes):
#     if global_idx is None:
#         raise gr.Error("No QA entry selected for this scan_id/split filter.")
#     idx = int(global_idx)

#     scene = load_scene(dataset_name, idx)
#     fig = build_plotly_figure(
#         scene=scene,
#         color_mode=color_mode,
#         point_size=point_size,
#         show_boxes=show_boxes,
#         show_axis=show_axis,
#         show_arrow=show_arrow,
#         axis_len=axis_len,
#         max_points=int(max_points),
#         max_boxes=int(max_boxes),
#     )
#     return fig

# # ---- Details panel toggle ----

# def toggle_details(is_open: bool):
#     new_state = not bool(is_open)
#     return gr.update(open=new_state), new_state

# def update_toggle_button_label(is_open: bool):
#     return gr.update(value=("Hide visualization details" if is_open else "Show visualization details"))

# # ---- Chat stub ----

# def answer_with_model(user_msg: str, dataset_name: str, global_idx, split_value: str):
#     user_msg = (user_msg or "").strip()
#     if not user_msg:
#         return ""

#     if global_idx is None:
#         return "No QA entry selected."

#     # (Right now) your inference code uses MSQAScanNet, so it only matches ScanNet scans.
#     if dataset_name != "scannet":
#         return "Model chat is currently wired for ScanNet only (MSQAScanNet). Switch to scannet to ask questions."

#     idx = int(global_idx)
#     qa = DATA_BY_DATASET[dataset_name][idx]
#     scene_id = qa["scan_id"]
#     situation = qa.get("situation", "")

#     scan_id = qa["scan_id"]

#     if split_value == "all":
#         effective_split = infer_split_from_scene(dataset_name, scan_id, qa_idx=global_idx)
#     else:
#         effective_split = split_value

#     try:
#         svc = get_msr3d_service()
#         svc.change_split(effective_split)  # ensure the model's dataset matches the default UI selection

#         # Lock to avoid concurrent generate() calls stepping on each other if Gradio queues multiple requests
#         with MSR3D_LOCK:
#             print(f"[model] Generating answer for dataset='{dataset_name}', scan_id='{scan_id}', split='{effective_split}', situation='{situation}' | user_msg='{user_msg}'")
#             ans = svc.answer(scene_id=scene_id, question=user_msg, situation=situation)
#         return ans
#     except Exception as e:
#         # show useful error without killing gradio
#         return f"[error] {type(e).__name__}: {e}"

# def chat_step(user_msg, history, dataset_name, global_idx, split_value):
#     history = history or []
#     user_msg = (user_msg or "").strip()
#     if not user_msg:
#         return "", history

#     # add user message
#     history.append({"role": "user", "content": user_msg})

#     # generate answer
#     model_answer = answer_with_model(user_msg, dataset_name, global_idx, split_value)

#     # add assistant message
#     history.append({"role": "assistant", "content": model_answer})

#     return "", history

# def clear_chat():
#     return []
# # ======================== Render Automations ================
# def on_dataset_change_and_render(dataset_name: str, color_mode, point_size,
#                                  show_boxes, show_axis, show_arrow, axis_len,
#                                  max_points, max_boxes):
#     scans = AVAILABLE_SCANS_BY_DATASET[dataset_name]
#     scan_val = scans[0] if scans else None
#     split_val = "all"

#     qa_choices = qa_choices_for_scan(dataset_name, scan_val, split_val) if scan_val else []
#     qa_val = qa_choices[0][1] if qa_choices else None

#     fig = None
#     if qa_val is not None:
#         fig = render(dataset_name, qa_val, color_mode, point_size,
#                      show_boxes, show_axis, show_arrow, axis_len, max_points, max_boxes)

#     return (
#         gr.update(choices=scans, value=scan_val),     # scan_id_dd
#         gr.update(value=split_val),                   # split_filter
#         gr.update(choices=qa_choices, value=qa_val),  # qa_dd
#         fig,                                          # plot
#     )


# def on_scan_or_split_change_and_render(dataset_name: str, scan_id: str, split_filter: str,
#                                        color_mode, point_size, show_boxes, show_axis,
#                                        show_arrow, axis_len, max_points, max_boxes):
#     choices = qa_choices_for_scan(dataset_name, scan_id, split_filter)
#     qa_val = choices[0][1] if choices else None

#     fig = None
#     if qa_val is not None:
#         fig = render(dataset_name, qa_val, color_mode, point_size,
#                      show_boxes, show_axis, show_arrow, axis_len, max_points, max_boxes)

#     return gr.update(choices=choices, value=qa_val), fig

# def infer_split_from_scene(dataset_name: str, scan_id: str, qa_idx: int | None = None) -> str:
#     """
#     Infer split for a scene (scan_id).
#     - If scan_id exists in exactly one split -> return it
#     - If it exists in multiple splits -> use the selected QA's split if provided
#     - Otherwise deterministic fallback (only for true ambiguity)
#     """
#     inds = SCAN_TO_INDICES_BY_DATASET[dataset_name].get(scan_id, [])
#     splits = sorted({DATA_BY_DATASET[dataset_name][i].get("split", "test") for i in inds})

#     if len(splits) == 1:
#         return splits[0]

#     if qa_idx is not None:
#         try:
#             return DATA_BY_DATASET[dataset_name][int(qa_idx)].get("split", "test")
#         except Exception:
#             pass

#     # Only used if the same scan_id exists in multiple splits and we can't disambiguate
#     for pref in ("test", "val", "train"):
#         if pref in splits:
#             return pref
#     return "test"

# # ======================== Gradio App ========================

# with gr.Blocks(
#     css="""
#     #scene-plot { height: 85vh !important; }
#     #scene-plot > div { height: 100% !important; }
#     """
# ) as demo:
#     gr.Markdown(
#         "## MSQA Multi-Dataset Scene Viewer (Gradio + Plotly)\n"
#         "Select **dataset → scene(folder) → QA**, optionally filter by split, then render.\n\n"
#         "**RScan** loads `<scan_id>/pcds.pth` inside each scan folder."
#     )

#     dataset_dd = gr.Dropdown(
#         choices=["scannet", "arkit", "rscan"],
#         value="scannet",
#         label="Dataset",
#         interactive=True,
#     )

#     with gr.Row():
#         scan_id_dd = gr.Dropdown(
#             choices=AVAILABLE_SCANS_BY_DATASET["scannet"],
#             value=AVAILABLE_SCANS_BY_DATASET["scannet"][0],
#             label="Scene (scan_id / folder name)",
#             interactive=True,
#         )

#         split_filter = gr.Dropdown(
#             choices=["all", "train", "val", "test"],
#             value="all",
#             label="Split filter",
#             interactive=True,
#         )

#         qa_dd = gr.Dropdown(
#             choices=[],
#             label="QA entry (within scene)",
#             interactive=True,
#         )

#     details_open = gr.State(False)
#     toggle_btn = gr.Button("Show visualization details")

#     with gr.Accordion("Visualization details", open=False) as details_panel:
#         with gr.Row():
#             color_mode = gr.Dropdown(choices=["RGB", "Instance"], value="RGB", label="Color mode")
#             point_size = gr.Slider(1, 10, value=2, step=1, label="Point size")

#         with gr.Row():
#             show_boxes = gr.Checkbox(value=False, label="Show instance bounding boxes")
#             show_axis = gr.Checkbox(value=False, label="Show world axis")
#             show_arrow = gr.Checkbox(value=True, label="Show situation arrow")
#             axis_len = gr.Slider(0.5, 5.0, value=1.5, step=0.1, label="Axis/arrow scale")

#         with gr.Row():
#             max_points = gr.Slider(10_000, 500_000, value=200_000, step=10_000, label="Max points (downsample for speed)")
#             max_boxes = gr.Slider(10, 500, value=200, step=10, label="Max boxes (cap for speed)")

#     with gr.Row():
#         with gr.Column(scale=7):
#             btn = gr.Button("Render")
#             plot = gr.Plot(elem_id="scene-plot", scale=5)

#         with gr.Column(scale=3):
#             gr.Markdown("### Ask your model about the scene")
#             chat = gr.Chatbot(label="Dialogue", height=400)
#             user_msg = gr.Textbox(label="Ask a question", placeholder="Ask about the scene...", lines=3)
#             with gr.Row():
#                 send = gr.Button("Send")
#                 clear = gr.Button("Clear")

#     # Init
#     #demo.load(fn=on_dataset_change, inputs=[dataset_dd], outputs=[scan_id_dd, split_filter, qa_dd])
#     demo.load(
#         fn=on_dataset_change_and_render,
#         inputs=[dataset_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_points, max_boxes],
#         outputs=[scan_id_dd, split_filter, qa_dd, plot],
#     )

#     # Dataset changes -> update scenes + QA list
#     dataset_dd.change(fn=on_dataset_change, inputs=[dataset_dd], outputs=[scan_id_dd, split_filter, qa_dd])
#     # dataset_dd.change(
#     #         fn=on_split_change,   # reuse the same logic
#     #         inputs=[dataset_dd, split_filter],
#     #         outputs=[scan_id_dd, qa_dd],
#     #     )

#     # Scan/split changes -> update QA list
#     scan_id_dd.change(fn=on_scan_or_split_change, inputs=[dataset_dd, scan_id_dd, split_filter], outputs=[qa_dd])
#     #split_filter.change(fn=on_scan_or_split_change, inputs=[dataset_dd, scan_id_dd, split_filter], outputs=[qa_dd])
#     split_filter.change(
#         fn=on_split_change,
#         inputs=[dataset_dd, split_filter],
#         outputs=[scan_id_dd, qa_dd],
#     )

#     # Details toggle
#     toggle_btn.click(
#         fn=toggle_details,
#         inputs=[details_open],
#         outputs=[details_panel, details_open],
#     ).then(
#         fn=update_toggle_button_label,
#         inputs=[details_open],
#         outputs=[toggle_btn],
#     )

#     # Render
#     btn.click(
#         fn=render,
#         inputs=[dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_points, max_boxes],
#         outputs=[plot],
#     )

#     # Chat
#     # send.click(fn=chat_step, inputs=[user_msg, chat, dataset_dd, qa_dd], outputs=[user_msg, chat])
#     # user_msg.submit(fn=chat_step, inputs=[user_msg, chat, dataset_dd, qa_dd], outputs=[user_msg, chat])
#     send.click(fn=chat_step, inputs=[user_msg, chat, dataset_dd, qa_dd, split_filter], outputs=[user_msg, chat])
#     user_msg.submit(fn=chat_step, inputs=[user_msg, chat, dataset_dd, qa_dd, split_filter], outputs=[user_msg, chat])

#     clear.click(fn=clear_chat, inputs=[], outputs=[chat])

# if __name__ == "__main__":
#     demo.launch(share=True)
#!/usr/bin/env python3
"""
MSQA Multi-Dataset Scene Viewer (Gradio + Plotly) — ScanNet + ARKitScenes + RScan (folder-based)

Upgrades in this version:
- Fast updates: points are rendered once per (dataset, scan_id, max_points) and cached.
- UI changes like color mode / point size / toggles update the existing figure (no disk reload).
- Heavy reload only happens when dataset/scan/max_points (or QA selection) changes.
- Fixes split/scene/QA “glitch”: split change chooses a scan that actually has QA in that split.
- Keeps your right-side chat stub and visualization accordion toggle.

Notes:
- Plotly in Gradio still sends a figure JSON each update, but we avoid torch.load/downsample/rebuild point trace.
"""

import os
import json
import torch
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import threading

# ======================== Config ========================

SCANNET_ROOT_DIR = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/text_annotations/msqa/scannet"
ARKIT_ROOT_DIR   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/text_annotations/msqa/arkitscenes"
RSCAN_ROOT_DIR   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/text_annotations/msqa/rscan"

SCANNET_JSON_PATHS = {
    "train": os.path.join(SCANNET_ROOT_DIR, "msqa_scannet_train.json"),
    "val":   os.path.join(SCANNET_ROOT_DIR, "msqa_scannet_val.json"),
    "test":  os.path.join(SCANNET_ROOT_DIR, "msqa_scannet_test.json"),
}
ARKIT_JSON_PATHS = {
    "train": os.path.join(ARKIT_ROOT_DIR, "msqa_arkitscenes_train.json"),
    "val":   os.path.join(ARKIT_ROOT_DIR, "msqa_arkitscenes_val.json"),
    "test":  os.path.join(ARKIT_ROOT_DIR, "msqa_arkitscenes_test.json"),
}
RSCAN_JSON_PATHS = {
    "train": os.path.join(RSCAN_ROOT_DIR, "msqa_rscan_train.json"),
    "val":   os.path.join(RSCAN_ROOT_DIR, "msqa_rscan_val.json"),
    "test":  os.path.join(RSCAN_ROOT_DIR, "msqa_rscan_test.json"),
}

SCANNET_PCD_ROOT = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment"
ARKIT_PCD_ROOT   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd-align/"

# RScan is folder-based and loads pcds.pth inside each scan folder
RSCAN_PCD_ROOT   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/rscan_base/3RScan-ours-align/"
RSCAN_PCD_FILE   = "pcds.pth"

ONLY_SHOW_SCANS_WITH_PTH = True

DATASET_SPECS = {
    "scannet": {"json_paths": SCANNET_JSON_PATHS, "pcd_root": SCANNET_PCD_ROOT},
    "arkit":   {"json_paths": ARKIT_JSON_PATHS,   "pcd_root": ARKIT_PCD_ROOT},
    "rscan":   {"json_paths": RSCAN_JSON_PATHS,   "pcd_root": RSCAN_PCD_ROOT},
}

# ======================== Model (chat stub) ========================

from msr3d.tools.interactive_service import MSR3DInteractiveService

MSR3D_EXPERIMENT_PATH = "MSR3D_BLIPT_PTv3_VIC_LORA_2"

MSR3D_SERVICE = None
MSR3D_LOCK = threading.Lock()

def get_msr3d_service():
    global MSR3D_SERVICE
    if MSR3D_SERVICE is None:
        MSR3D_SERVICE = MSR3DInteractiveService(
            experiment_path=MSR3D_EXPERIMENT_PATH,
            split="test",
        )
    return MSR3D_SERVICE

# ======================== Utils ========================

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def ensure_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def normalize_rgb01(colors):
    """
    Accept colors in:
      - [-1,1] float
      - [0,1] float
      - [0,255] float/uint8
    Return float RGB in [0,1].
    """
    c = ensure_np(colors).astype(np.float32).reshape(-1, 3)
    cmin, cmax = float(np.nanmin(c)), float(np.nanmax(c))

    if cmin >= -1.01 and cmax <= 1.01:
        # either [-1,1] or [0,1]
        if cmin < 0.0:
            return np.clip((c + 1.0) * 0.5, 0.0, 1.0)
        return np.clip(c, 0.0, 1.0)

    # assume [0,255]
    return np.clip(c / 255.0, 0.0, 1.0)

def hash_colors_for_labels(labels, seed=0):
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)
    table = {}
    for u in uniq:
        col = rng.random(3) * 0.8 + 0.2
        table[int(u)] = col.astype(np.float32)
    return np.array([table[int(x)] for x in labels], dtype=np.float32)

# -------- Orientation handling --------

def get_view_vector_from_orientation(orientation):
    o = ensure_np(orientation).astype(np.float32).reshape(-1)

    if o.size == 4:
        yaw = R.from_quat(o).as_euler("xyz", degrees=False)[-1]
        d = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        return d / (np.linalg.norm(d) + 1e-12)

    if o.size == 3:
        n = float(np.linalg.norm(o))
        if 0.5 <= n <= 1.5:
            return (o / (n + 1e-12)).astype(np.float32)
        yaw = float(o[-1])
        d = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        return d / (np.linalg.norm(d) + 1e-12)

    return np.array([1.0, 0.0, 0.0], dtype=np.float32)

# ======================== Plotly overlay geometry ========================

def aabb_edges(minb, maxb):
    minb = np.asarray(minb, dtype=np.float32)
    maxb = np.asarray(maxb, dtype=np.float32)

    x0, y0, z0 = minb
    x1, y1, z1 = maxb

    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float32)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [(corners[i], corners[j]) for i, j in edges]

def build_instance_bboxes_as_traces(xyz, instance_labels, max_boxes=200, seed=123):
    if instance_labels is None:
        return []

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    inst = np.asarray(instance_labels).reshape(-1)
    uniq = np.unique(inst)

    uniq_cols = hash_colors_for_labels(uniq, seed=seed)
    col_tab = {int(u): uniq_cols[i] for i, u in enumerate(uniq)}

    traces = []
    if len(uniq) > max_boxes:
        uniq = uniq[:max_boxes]

    for u in uniq:
        mask = (inst == u)
        if not np.any(mask):
            continue
        pts = xyz[mask]
        if pts.shape[0] < 2:
            continue

        minb = pts.min(0)
        maxb = pts.max(0)
        segs = aabb_edges(minb, maxb)
        col = col_tab[int(u)]

        xs, ys, zs = [], [], []
        for p0, p1 in segs:
            xs += [p0[0], p1[0], None]
            ys += [p0[1], p1[1], None]
            zs += [p0[2], p1[2], None]

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=4, color=f"rgb({int(col[0]*255)},{int(col[1]*255)},{int(col[2]*255)})"),
            showlegend=False,
        ))

    return traces

def make_world_axis_traces(origin, axis_len=1.0):
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    O = origin
    X = origin + np.array([axis_len, 0, 0], dtype=np.float32)
    Y = origin + np.array([0, axis_len, 0], dtype=np.float32)
    Z = origin + np.array([0, 0, axis_len], dtype=np.float32)

    def line(p0, p1, color):
        return go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode="lines",
            line=dict(width=8, color=color),
            showlegend=False,
        )

    return [line(O, X, "red"), line(O, Y, "green"), line(O, Z, "blue")]

def make_situation_arrow_trace(location, orientation, scale=0.8):
    loc = np.asarray(location, dtype=np.float32).reshape(3)
    d = get_view_vector_from_orientation(orientation)

    loc2 = loc.copy()
    loc2[2] += float(0.15)
    tip = loc2 + d * float(scale)

    return go.Scatter3d(
        x=[loc2[0], tip[0]],
        y=[loc2[1], tip[1]],
        z=[loc2[2], tip[2]],
        mode="lines+markers",
        line=dict(width=10, color="orange"),
        marker=dict(size=4),
        showlegend=False,
    )

# ======================== Dataset path resolution ========================

def resolve_pth_path(dataset_name: str, scan_id: str) -> str:
    """
    - RScan: <RSCAN_PCD_ROOT>/<scan_id>/pcds.pth
    - Others: <PCD_ROOT>/<scan_id>.pth
    """
    if dataset_name == "rscan":
        return os.path.join(RSCAN_PCD_ROOT, scan_id, RSCAN_PCD_FILE)
    root = DATASET_SPECS[dataset_name]["pcd_root"]
    return os.path.join(root, f"{scan_id}.pth")

def pth_exists(dataset_name: str, scan_id: str) -> bool:
    return os.path.exists(resolve_pth_path(dataset_name, scan_id))

# ======================== Load + merge splits (per dataset) ========================

DATA_BY_DATASET = {}
SCAN_TO_INDICES_BY_DATASET = {}
AVAILABLE_SCANS_BY_DATASET = {}

def build_scan_index(data):
    scan_to_indices = defaultdict(list)
    for i, qa in enumerate(data):
        scan_to_indices[qa["scan_id"]].append(i)
    return dict(scan_to_indices)

for dname, spec in DATASET_SPECS.items():
    data = []
    for split_name, path in spec["json_paths"].items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing JSON for dataset '{dname}' split '{split_name}': {path}")
        items = load_json(path)
        for it in items:
            it = dict(it)
            it["split"] = split_name
            data.append(it)

    DATA_BY_DATASET[dname] = data
    SCAN_TO_INDICES_BY_DATASET[dname] = build_scan_index(data)

    all_scans = sorted(SCAN_TO_INDICES_BY_DATASET[dname].keys())
    if ONLY_SHOW_SCANS_WITH_PTH:
        avail = [sid for sid in all_scans if pth_exists(dname, sid)]
    else:
        avail = all_scans

    if len(avail) == 0:
        raise RuntimeError(f"No scan_ids available for dataset '{dname}'. Check JSON paths and PCD roots.")
    AVAILABLE_SCANS_BY_DATASET[dname] = avail

# ======================== QA dropdown helpers ========================

def qa_label(dataset_name: str, i: int) -> str:
    qa = DATA_BY_DATASET[dataset_name][i]
    sit = (qa.get("situation", "") or "").strip().replace("\n", " ")
    if len(sit) > 90:
        sit = sit[:87] + "..."
    return f"{i} | {qa['scan_id']} | {qa.get('split','?')} | {sit}"

def qa_choices_for_scan(dataset_name: str, scan_id: str, split_filter: str):
    inds = SCAN_TO_INDICES_BY_DATASET[dataset_name].get(scan_id, [])
    if split_filter != "all":
        inds = [i for i in inds if DATA_BY_DATASET[dataset_name][i].get("split") == split_filter]
    return [(qa_label(dataset_name, i), i) for i in inds]

# ======================== Scene load (heavy) ========================

def load_scene_full(dataset_name: str, idx: int):
    qa = DATA_BY_DATASET[dataset_name][idx]
    scan_id = qa["scan_id"]

    pth_path = resolve_pth_path(dataset_name, scan_id)
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Missing PTH: {pth_path}")

    pcd_data = torch.load(pth_path, weights_only=False)
    if not isinstance(pcd_data, (tuple, list)) or len(pcd_data) < 2:
        raise ValueError(f"Unsupported PTH format for {pth_path}: {type(pcd_data)}")

    points = ensure_np(pcd_data[0]).astype(np.float32).reshape(-1, 3)
    colors = ensure_np(pcd_data[1]).astype(np.float32).reshape(-1, 3)
    rgb01 = normalize_rgb01(colors)

    instance_labels = None
    if len(pcd_data) >= 3:
        cand_last = ensure_np(pcd_data[-1]).reshape(-1)
        if cand_last.shape[0] == points.shape[0]:
            instance_labels = cand_last.astype(np.int32)
        else:
            cand2 = ensure_np(pcd_data[2]).reshape(-1)
            if cand2.shape[0] == points.shape[0]:
                instance_labels = cand2.astype(np.int32)

    location = qa.get("location", None)
    orientation = qa.get("orientation", None)

    if isinstance(location, dict):
        location = [location.get("x", 0.0), location.get("y", 0.0), location.get("z", 0.0)]

    if isinstance(orientation, dict):
        if all(k in orientation for k in ["_x", "_y", "_z", "_w"]):
            orientation = [orientation["_x"], orientation["_y"], orientation["_z"], orientation["_w"]]
        elif all(k in orientation for k in ["x", "y", "z"]):
            orientation = [orientation["x"], orientation["y"], orientation["z"]]

    return {
        "scan_id": scan_id,
        "split": qa.get("split", "unknown"),
        "situation": qa.get("situation", ""),
        "points": points,
        "rgb01": rgb01,
        "instance_labels": instance_labels,
        "location": location,
        "orientation": orientation,
    }

# ======================== Cache: downsampled per (dataset, scan_id, max_points) ========================

SCENE_CACHE = {}  # key -> cached dict

def get_cached_scene(dataset_name: str, idx: int, max_points: int):
    qa = DATA_BY_DATASET[dataset_name][idx]
    scan_id = qa["scan_id"]
    key = (dataset_name, scan_id, int(max_points))

    if key in SCENE_CACHE:
        return SCENE_CACHE[key], key

    scene = load_scene_full(dataset_name, idx)
    xyz = scene["points"]
    rgb = scene["rgb01"]
    inst = scene["instance_labels"]

    N = xyz.shape[0]
    if max_points is not None and int(max_points) > 0 and N > int(max_points):
        sel = np.random.default_rng(0).choice(N, size=int(max_points), replace=False)
        xyz = xyz[sel]
        rgb = rgb[sel]
        inst = inst[sel] if inst is not None else None

    cached = {
        "scan_id": scene["scan_id"],
        "split": scene["split"],
        "situation": scene["situation"],
        "xyz": xyz.astype(np.float32),
        "rgb01": rgb.astype(np.float32),
        "inst": inst.astype(np.int32) if inst is not None else None,
        "location": scene["location"],
        "orientation": scene["orientation"],
        "center": xyz.mean(axis=0).astype(np.float32),
    }
    SCENE_CACHE[key] = cached
    return cached, key

# ======================== Plotly figure: base + style overlays ========================

def make_base_figure(cached, point_size: float):
    xyz = cached["xyz"]
    rgb01 = cached["rgb01"]

    cols255 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    color_str = [f"rgb({r},{g},{b})" for r, g, b in cols255]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode="markers",
        marker=dict(size=float(point_size), color=color_str, opacity=1.0),
        showlegend=False,
    ))

    title = f"{cached['scan_id']} | split: {cached.get('split','?')}"
    if cached.get("situation"):
        title += f" | {cached['situation']}"

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="black",
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
    )
    return fig

def apply_style_and_overlays(fig, cached, color_mode: str, point_size: float,
                             show_boxes: bool, show_axis: bool, show_arrow: bool,
                             axis_len: float, max_boxes: int):
    # update marker size
    fig.data[0].marker.size = float(point_size)

    # update marker color
    if color_mode == "RGB":
        cols = cached["rgb01"]
    elif color_mode == "Instance":
        cols = cached["rgb01"] if cached["inst"] is None else hash_colors_for_labels(cached["inst"], seed=123)
    else:
        cols = cached["rgb01"]

    cols255 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    fig.data[0].marker.color = [f"rgb({r},{g},{b})" for r, g, b in cols255]

    # clear overlays (keep points trace)
    fig.data = fig.data[:1]

    # add overlays
    if show_boxes and cached["inst"] is not None:
        for t in build_instance_bboxes_as_traces(cached["xyz"], cached["inst"], max_boxes=int(max_boxes)):
            fig.add_trace(t)

    if show_axis:
        for t in make_world_axis_traces(cached["center"], axis_len=float(axis_len)):
            fig.add_trace(t)

    if show_arrow and cached["location"] is not None and cached["orientation"] is not None:
        try:
            fig.add_trace(make_situation_arrow_trace(cached["location"], cached["orientation"], scale=float(axis_len)))
        except Exception as e:
            print(f"[warn] could not render situation arrow: {e}")

    return fig

# ======================== Split inference (for chat) ========================

def infer_split_from_scene(dataset_name: str, scan_id: str, qa_idx: int | None = None) -> str:
    inds = SCAN_TO_INDICES_BY_DATASET[dataset_name].get(scan_id, [])
    splits = sorted({DATA_BY_DATASET[dataset_name][i].get("split", "test") for i in inds})

    if len(splits) == 1:
        return splits[0]

    if qa_idx is not None:
        try:
            return DATA_BY_DATASET[dataset_name][int(qa_idx)].get("split", "test")
        except Exception:
            pass

    for pref in ("test", "val", "train"):
        if pref in splits:
            return pref
    return "test"

# ======================== Gradio callbacks: dataset/split/scan/qa ========================

def scans_for_split(dataset_name: str, split_filter: str):
    data = DATA_BY_DATASET[dataset_name]
    scan_to_inds = SCAN_TO_INDICES_BY_DATASET[dataset_name]

    if split_filter == "all":
        scans = list(scan_to_inds.keys())
    else:
        scans = []
        for sid, inds in scan_to_inds.items():
            if any(data[i].get("split") == split_filter for i in inds):
                scans.append(sid)

    scans = sorted(scans)

    if ONLY_SHOW_SCANS_WITH_PTH:
        scans = [sid for sid in scans if pth_exists(dataset_name, sid)]

    return scans

def on_split_change(dataset_name: str, split_filter: str):
    scans = scans_for_split(dataset_name, split_filter)

    scan_val = None
    qa_choices = []
    qa_val = None

    # pick first scan that actually has QA choices
    for sid in scans:
        choices = qa_choices_for_scan(dataset_name, sid, split_filter)
        if choices:
            scan_val = sid
            qa_choices = choices
            qa_val = choices[0][1]
            break

    return (
        gr.update(choices=scans, value=scan_val),
        gr.update(choices=qa_choices, value=qa_val),
    )

def on_dataset_change(dataset_name: str):
    scans = AVAILABLE_SCANS_BY_DATASET[dataset_name]
    split_val = "all"

    scan_val = None
    qa_choices = []
    qa_val = None

    for sid in scans:
        choices = qa_choices_for_scan(dataset_name, sid, split_val)
        if choices:
            scan_val = sid
            qa_choices = choices
            qa_val = choices[0][1]
            break

    return (
        gr.update(choices=scans, value=scan_val),
        gr.update(value=split_val),
        gr.update(choices=qa_choices, value=qa_val),
    )

def on_scan_change(dataset_name: str, scan_id: str, split_filter: str):
    choices = qa_choices_for_scan(dataset_name, scan_id, split_filter)
    qa_val = choices[0][1] if choices else None
    return gr.update(choices=choices, value=qa_val)

# ======================== Render callbacks: base vs style ========================

def build_base(dataset_name, global_idx, max_points, point_size):
    if global_idx is None:
        raise gr.Error("No QA entry selected.")
    idx = int(global_idx)

    cached, key = get_cached_scene(dataset_name, idx, int(max_points))
    fig = make_base_figure(cached, float(point_size))
    return fig, fig, key  # plot, fig_state, key_state

def update_style(fig, key, dataset_name, global_idx,
                 color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points):
    if global_idx is None:
        return fig, fig, key

    idx = int(global_idx)
    cached, new_key = get_cached_scene(dataset_name, idx, int(max_points))

    # if missing or key mismatch, rebuild base defensively
    if fig is None or key != new_key:
        fig = make_base_figure(cached, float(point_size))
        key = new_key

    fig = apply_style_and_overlays(
        fig=fig,
        cached=cached,
        color_mode=color_mode,
        point_size=point_size,
        show_boxes=show_boxes,
        show_axis=show_axis,
        show_arrow=show_arrow,
        axis_len=axis_len,
        max_boxes=max_boxes,
    )
    return fig, fig, key  # plot, fig_state, key_state

# ======================== Details panel toggle ========================

def toggle_details(is_open: bool):
    new_state = not bool(is_open)
    return gr.update(open=new_state), new_state

def update_toggle_button_label(is_open: bool):
    return gr.update(value=("Hide visualization details" if is_open else "Show visualization details"))

# ======================== Chat stub ========================

def answer_with_model(user_msg: str, dataset_name: str, global_idx, split_value: str):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return ""

    if global_idx is None:
        return "No QA entry selected."

    if dataset_name != "scannet":
        return "Model chat is currently wired for ScanNet only (MSQAScanNet). Switch to scannet to ask questions."

    idx = int(global_idx)
    qa = DATA_BY_DATASET[dataset_name][idx]
    scene_id = qa["scan_id"]
    situation = qa.get("situation", "")

    scan_id = qa["scan_id"]
    if split_value == "all":
        effective_split = infer_split_from_scene(dataset_name, scan_id, qa_idx=global_idx)
    else:
        effective_split = split_value

    try:
        svc = get_msr3d_service()
        svc.change_split(effective_split)

        with MSR3D_LOCK:
            print(f"[model] Generating answer for dataset='{dataset_name}', scan_id='{scan_id}', split='{effective_split}', situation='{situation}' | user_msg='{user_msg}'")
            ans = svc.answer(scene_id=scene_id, question=user_msg, situation=situation)
        return ans
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"

def chat_step(user_msg, history, dataset_name, global_idx, split_value):
    history = history or []
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return "", history

    history.append({"role": "user", "content": user_msg})
    model_answer = answer_with_model(user_msg, dataset_name, global_idx, split_value)
    history.append({"role": "assistant", "content": model_answer})

    return "", history

def clear_chat():
    return []

# ======================== Gradio App ========================

with gr.Blocks(
    css="""
    #scene-plot { height: 85vh !important; }
    #scene-plot > div { height: 100% !important; }
    """
) as demo:
    gr.Markdown(
        "## MSQA Multi-Dataset Scene Viewer (Gradio + Plotly)\n"
        "Select **dataset → scene(folder) → QA**, optionally filter by split.\n\n"
        "**RScan** loads `<scan_id>/pcds.pth` inside each scan folder.\n\n"
        "Fast mode: base geometry cached; styling updates avoid disk reload."
    )

    dataset_dd = gr.Dropdown(
        choices=["scannet", "arkit", "rscan"],
        value="scannet",
        label="Dataset",
        interactive=True,
    )

    with gr.Row():
        scan_id_dd = gr.Dropdown(
            choices=AVAILABLE_SCANS_BY_DATASET["scannet"],
            value=AVAILABLE_SCANS_BY_DATASET["scannet"][0],
            label="Scene (scan_id / folder name)",
            interactive=True,
        )

        split_filter = gr.Dropdown(
            choices=["all", "train", "val", "test"],
            value="all",
            label="Split filter",
            interactive=True,
        )

        qa_dd = gr.Dropdown(
            choices=[],
            label="QA entry (within scene)",
            interactive=True,
        )

    # states for accordion + fast plot updates
    details_open = gr.State(False)
    fig_state = gr.State(None)
    key_state = gr.State(None)

    toggle_btn = gr.Button("Show visualization details")

    with gr.Accordion("Visualization details", open=False) as details_panel:
        with gr.Row():
            color_mode = gr.Dropdown(choices=["RGB", "Instance"], value="RGB", label="Color mode")
            point_size = gr.Slider(1, 10, value=2, step=1, label="Point size")

        with gr.Row():
            show_boxes = gr.Checkbox(value=False, label="Show instance bounding boxes")
            show_axis = gr.Checkbox(value=False, label="Show world axis")
            show_arrow = gr.Checkbox(value=True, label="Show situation arrow")
            axis_len = gr.Slider(0.5, 5.0, value=1.5, step=0.1, label="Axis/arrow scale")

        with gr.Row():
            max_points = gr.Slider(10_000, 500_000, value=200_000, step=10_000, label="Max points (downsample for speed)")
            max_boxes = gr.Slider(10, 500, value=200, step=10, label="Max boxes (cap for speed)")

    with gr.Row():
        with gr.Column(scale=7):
            btn = gr.Button("Render")
            plot = gr.Plot(elem_id="scene-plot", scale=5)

        with gr.Column(scale=3):
            gr.Markdown("### Ask your model about the scene")
            chat = gr.Chatbot(label="Dialogue", height=400)
            user_msg = gr.Textbox(label="Ask a question", placeholder="Ask about the scene...", lines=3)
            with gr.Row():
                send = gr.Button("Send")
                clear = gr.Button("Clear")

    # ============ Init ============
    # Load dropdowns (dataset default) then render base+style once.
    demo.load(fn=on_dataset_change, inputs=[dataset_dd], outputs=[scan_id_dd, split_filter, qa_dd]).then(
        fn=build_base,
        inputs=[dataset_dd, qa_dd, max_points, point_size],
        outputs=[plot, fig_state, key_state],
    ).then(
        fn=update_style,
        inputs=[fig_state, key_state, dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points],
        outputs=[plot, fig_state, key_state],
    )

    # ============ Dataset changes ============
    # Update dropdowns only; do not auto-render unless you want it.
    dataset_dd.change(fn=on_dataset_change, inputs=[dataset_dd], outputs=[scan_id_dd, split_filter, qa_dd])

    # ============ Split changes ============
    split_filter.change(fn=on_split_change, inputs=[dataset_dd, split_filter], outputs=[scan_id_dd, qa_dd])

    # ============ Scan changes ============
    scan_id_dd.change(fn=on_scan_change, inputs=[dataset_dd, scan_id_dd, split_filter], outputs=[qa_dd])

    # ============ Details toggle ============
    toggle_btn.click(
        fn=toggle_details,
        inputs=[details_open],
        outputs=[details_panel, details_open],
    ).then(
        fn=update_toggle_button_label,
        inputs=[details_open],
        outputs=[toggle_btn],
    )

    # ============ Render button (heavy then light) ============
    btn.click(
        fn=build_base,
        inputs=[dataset_dd, qa_dd, max_points, point_size],
        outputs=[plot, fig_state, key_state],
    ).then(
        fn=update_style,
        inputs=[fig_state, key_state, dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points],
        outputs=[plot, fig_state, key_state],
    )

    # ============ Light updates (no disk reload) ============
    # These update the cached figure in-place.
    color_mode.change(
        fn=update_style,
        inputs=[fig_state, key_state, dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points],
        outputs=[plot, fig_state, key_state],
    )
    point_size.change(
        fn=update_style,
        inputs=[fig_state, key_state, dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points],
        outputs=[plot, fig_state, key_state],
    )
    show_boxes.change(
        fn=update_style,
        inputs=[fig_state, key_state, dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points],
        outputs=[plot, fig_state, key_state],
    )
    show_axis.change(
        fn=update_style,
        inputs=[fig_state, key_state, dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points],
        outputs=[plot, fig_state, key_state],
    )
    show_arrow.change(
        fn=update_style,
        inputs=[fig_state, key_state, dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points],
        outputs=[plot, fig_state, key_state],
    )
    axis_len.change(
        fn=update_style,
        inputs=[fig_state, key_state, dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points],
        outputs=[plot, fig_state, key_state],
    )
    max_boxes.change(
        fn=update_style,
        inputs=[fig_state, key_state, dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points],
        outputs=[plot, fig_state, key_state],
    )

    # Max points is a heavy trigger: rebuild base.
    max_points.change(
        fn=build_base,
        inputs=[dataset_dd, qa_dd, max_points, point_size],
        outputs=[plot, fig_state, key_state],
    ).then(
        fn=update_style,
        inputs=[fig_state, key_state, dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_boxes, max_points],
        outputs=[plot, fig_state, key_state],
    )

    # ============ Chat ============
    send.click(fn=chat_step, inputs=[user_msg, chat, dataset_dd, qa_dd, split_filter], outputs=[user_msg, chat])
    user_msg.submit(fn=chat_step, inputs=[user_msg, chat, dataset_dd, qa_dd, split_filter], outputs=[user_msg, chat])
    clear.click(fn=clear_chat, inputs=[], outputs=[chat])

if __name__ == "__main__":
    demo.launch(share=True)
