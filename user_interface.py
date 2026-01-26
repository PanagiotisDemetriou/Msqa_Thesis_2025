# #!/usr/bin/env python3
# """
# MSQA ScanNet Scene Viewer (Gradio + Plotly) — train/val/test merged (Drop-in)

# - Loads MSQA ScanNet JSON for train + val + test, merges into a single DATA list.
# - Scene dropdown includes all scan_ids across splits (optionally only those with .pth on disk).
# - QA dropdown lists only entries for the selected scan_id (and shows split in the label).

# UI additions:
# 1) Visualization details panel hidden by default (Accordion) + Show/Hide toggle button.
# 2) A right-side "Situation-aware QA" chat panel (stub) for your future model.
#    - Chat receives the selected QA/global_idx context.
#    - Replace `answer_with_model()` with your model inference.
# """

# import os
# import json
# import torch
# import numpy as np
# import gradio as gr
# import plotly.graph_objects as go
# from scipy.spatial.transform import Rotation as R
# from collections import defaultdict

# # ======================== Config ========================

# SCANNET_ROOT_DIR = "/mnt/d/Thesis/data/text_annotations/msqa/scannet"
# ARKIT_ROOT_DIR = "/mnt/d/Thesis/data/text_annotations/msqa/arkitscenes"
# RSCAN_ROOT_DIR = "/mnt/d/Thesis/data/text_annotations/msqa/rscan"

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

# SCANNET_PCD_ROOT = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment"
# ARKIT_PCD_ROOT = "/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd-align/pcd-align"
# RSCAN_PCD_ROOT = "/mnt/d/Thesis/data/MSR3D_v2_pcds/rscan_base/3RScan-ours-align/3RScan-ours-align"

# # If True: dropdown only shows scans that have a corresponding .pth in PCD_ROOT
# ONLY_SHOW_SCANS_WITH_PTH = True

# # ======================== Utils ========================

# def load_json(path: str):
#     with open(path, "r") as f:
#         return json.load(f)

# def ensure_np(x):
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)

# def to_float_colors(rgb_uint8):
#     return np.asarray(rgb_uint8, dtype=np.float32) / 255.0

# # Official ScanNet20 palette
# SCANNET20_COLORS = to_float_colors(np.array([
#     [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120],
#     [188, 189, 34], [140, 86, 75], [255, 152, 150], [214, 39, 40],
#     [197, 176, 213], [148, 103, 189], [196, 156, 148], [23, 190, 207],
#     [247, 182, 210], [219, 219, 141], [255, 127, 14], [158, 218, 229],
#     [44, 160, 44], [112, 128, 144], [227, 119, 194], [82, 84, 163],
# ], dtype=np.uint8))

# def hash_colors_for_labels(labels, seed=0):
#     labels = np.asarray(labels).reshape(-1)
#     rng = np.random.default_rng(seed)
#     uniq = np.unique(labels)
#     table = {}
#     for u in uniq:
#         col = rng.random(3) * 0.8 + 0.2
#         table[int(u)] = col.astype(np.float32)
#     return np.array([table[int(x)] for x in labels], dtype=np.float32)

# def palette_colors_for_labels(labels, palette):
#     labels = np.asarray(labels).reshape(-1)
#     uniq = np.unique(labels)
#     table = {}
#     K = len(palette)
#     for i, u in enumerate(uniq):
#         table[int(u)] = palette[i % K]
#     return np.array([table[int(x)] for x in labels], dtype=np.float32)

# # -------- Robust orientation handling --------

# def get_view_vector_from_orientation(orientation):
#     """
#     Accept orientation as:
#       - quaternion (x,y,z,w) -> use yaw (euler z)
#       - euler xyz (3,) radians -> use yaw = z
#       - direction vector (3,) -> use directly (normalized)
#     Returns normalized direction vector (3,).
#     """
#     o = ensure_np(orientation).astype(np.float32).reshape(-1)

#     if o.size == 4:
#         yaw = R.from_quat(o).as_euler("xyz", degrees=False)[-1]
#         d = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
#         return d / (np.linalg.norm(d) + 1e-12)

#     if o.size == 3:
#         n = float(np.linalg.norm(o))
#         if 0.5 <= n <= 1.5:  # direction vector
#             return (o / (n + 1e-12)).astype(np.float32)
#         yaw = float(o[-1])  # euler yaw
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
#     loc2[2] += float(0.15)  # lift above surfaces

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

# # ======================== Scene load + render ========================

# def load_scene_by_index(data, idx: int):
#     qa = data[idx]
#     scan_id = qa["scan_id"]
#     pth_path = os.path.join(SCANNET_PCD_ROOT, f"{scan_id}.pth")
#     if not os.path.exists(pth_path):
#         raise FileNotFoundError(f"Missing PTH: {pth_path}")

#     pcd_data = torch.load(pth_path, weights_only=False)

#     points = ensure_np(pcd_data[0]).astype(np.float32)
#     colors = ensure_np(pcd_data[1]).astype(np.float32)

#     colors = colors / 127.5 - 1.0
#     rgb01 = np.clip((colors + 1.0) * 0.5, 0.0, 1.0)

#     instance_labels = ensure_np(pcd_data[-1]).astype(np.int32)

#     # placeholders for future
#     segment20 = None
#     segment200 = None

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
#         "segment20": segment20,
#         "segment200": segment200,
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
#         seg20_vis = scene["segment20"][idx] if scene["segment20"] is not None else None
#         seg200_vis = scene["segment200"][idx] if scene["segment200"] is not None else None
#     else:
#         xyz_vis = xyz
#         rgb_vis = scene["rgb01"]
#         inst_vis = scene["instance_labels"]
#         seg20_vis = scene["segment20"]
#         seg200_vis = scene["segment200"]

#     if color_mode == "RGB":
#         cols = rgb_vis
#     elif color_mode == "Instance":
#         cols = rgb_vis if inst_vis is None else hash_colors_for_labels(inst_vis, seed=123)
#     elif color_mode == "Segment20":
#         cols = rgb_vis if seg20_vis is None else palette_colors_for_labels(seg20_vis, SCANNET20_COLORS)
#     elif color_mode == "Segment200":
#         cols = rgb_vis if seg200_vis is None else hash_colors_for_labels(seg200_vis, seed=200)
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

#     title = f"ScanNet Scene: {scene['scan_id']} | split: {scene.get('split','?')}"
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

# # ======================== Load + merge splits ========================

# DATA = []
# for split_name, path in SCANNET_JSON_PATHS.items():
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Missing JSON for split '{split_name}': {path}")
#     items = load_json(path)
#     for it in items:
#         it = dict(it)  # shallow copy
#         it["split"] = split_name
#         DATA.append(it)

# # ======================== Indexing for dropdowns ========================

# def build_scan_index(data):
#     scan_to_indices = defaultdict(list)
#     for i, qa in enumerate(data):
#         scan_to_indices[qa["scan_id"]].append(i)
#     return dict(scan_to_indices)

# SCAN_TO_INDICES = build_scan_index(DATA)

# def pth_exists_for_scan(scan_id: str) -> bool:
#     return os.path.exists(os.path.join(SCANNET_PCD_ROOT, f"{scan_id}.pth"))

# ALL_SCANS = sorted(SCAN_TO_INDICES.keys())
# if ONLY_SHOW_SCANS_WITH_PTH:
#     AVAILABLE_SCANS = [sid for sid in ALL_SCANS if pth_exists_for_scan(sid)]
# else:
#     AVAILABLE_SCANS = ALL_SCANS

# if len(AVAILABLE_SCANS) == 0:
#     raise RuntimeError("No scan_ids available. Check JSON_PATHS and/or PCD_ROOT.")

# def qa_label(i: int) -> str:
#     qa = DATA[i]
#     sit = (qa.get("situation", "") or "").strip().replace("\n", " ")
#     if len(sit) > 90:
#         sit = sit[:87] + "..."
#     return f"{i} | {qa['scan_id']} | {qa.get('split','?')} | {sit}"

# def qa_choices_for_scan(scan_id: str, split_filter: str):
#     inds = SCAN_TO_INDICES.get(scan_id, [])
#     if split_filter != "all":
#         inds = [i for i in inds if DATA[i].get("split") == split_filter]
#     return [(qa_label(i), i) for i in inds]

# # ======================== Gradio callbacks ========================

# def on_scan_or_split_change(scan_id: str, split_filter: str):
#     choices = qa_choices_for_scan(scan_id, split_filter)
#     default_val = choices[0][1] if choices else None
#     return gr.update(choices=choices, value=default_val)

# def render(global_idx, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_points, max_boxes):
#     if global_idx is None:
#         raise gr.Error("No QA entry selected for this scan_id/split filter.")
#     idx = int(global_idx)

#     scene = load_scene_by_index(DATA, idx)
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

#     info = (
#         f"global_idx: {idx}\n"
#         f"split: {scene.get('split','')}\n"
#         f"scan_id: {scene['scan_id']}\n"
#         f"situation: {scene.get('situation','')}\n"
#         f"points: {scene['points'].shape[0]}\n"
#         f"mode: {color_mode}\n"
#     )
#     #return fig, info
#     return fig

# # ---- Details panel toggle ----

# def toggle_details(is_open: bool):
#     new_state = not bool(is_open)
#     return gr.update(open=new_state), new_state

# def update_toggle_button_label(is_open: bool):
#     return gr.update(value=("Hide visualization details" if is_open else "Show visualization details"))

# # ---- Situation-aware QA panel (stub) ----

# def get_context_for_idx(global_idx):
#     if global_idx is None:
#         return {"ok": False, "error": "No QA entry selected."}
#     idx = int(global_idx)
#     qa = DATA[idx]
#     return {
#         "ok": True,
#         "global_idx": idx,
#         "scan_id": qa.get("scan_id"),
#         "split": qa.get("split"),
#         "situation": qa.get("situation", ""),
#         "question": qa.get("question", ""),
#         "answers": qa.get("answers", qa.get("answer", "")),
#         "location": qa.get("location", None),
#         "orientation": qa.get("orientation", None),
#     }

# def update_context_box(global_idx):
#     ctx = get_context_for_idx(global_idx)
#     if not ctx.get("ok"):
#         return ctx.get("error", "No QA entry selected.")
#     return (
#         f"global_idx: {ctx['global_idx']}\n"
#         f"scan_id: {ctx['scan_id']}\n"
#         f"split: {ctx['split']}\n"
#         f"situation: {ctx['situation']}\n"
#         f"location: {ctx['location']}\n"
#         f"orientation: {ctx['orientation']}\n"
#     )

# def answer_with_model(user_msg: str, global_idx):
#     """
#     Replace this function with your real situation-aware model call.
#     It already receives the selected QA/global_idx for context.
#     """
#     user_msg = (user_msg or "").strip()
#     if not user_msg:
#         return ""

#     ctx = get_context_for_idx(global_idx)
#     if not ctx.get("ok"):
#         return f"[error] {ctx.get('error')}"

#     # -------- STUB RESPONSE (replace) --------
#     return (
#         "Model stub (replace `answer_with_model()` with your model inference).\n\n"
#         f"Context:\n"
#         f"- scan_id: {ctx['scan_id']}\n"
#         f"- split: {ctx['split']}\n"
#         f"- situation: {ctx['situation']}\n\n"
#         f"User question:\n{user_msg}"
#     )

# def chat_step(user_msg, history, global_idx):
#     history = history or []
#     user_msg = (user_msg or "").strip()
#     if not user_msg:
#         return "", history

#     model_answer = answer_with_model(user_msg, global_idx)
#     history.append((user_msg, model_answer))
#     return "", history

# def clear_chat():
#     return []

# # ======================== Gradio App ========================

# with gr.Blocks(
#     css="""
#     #scene-plot { height: 85vh !important; }
#     #scene-plot > div { height: 100% !important; }
#     """
# ) as demo:
#     gr.Markdown(
#         "## MSQA ScanNet Scene Viewer (Gradio + Plotly)\n"
#         "Loads **train/val/test** MSQA JSONs. Select a **scene**, optionally filter by **split**, then render.\n\n"
#         "Visualization details are hidden by default; use the button to expand/collapse.\n"
#         "A right-side chat panel is included as a stub for your future situation-aware model."
#     )

#     with gr.Row():
#         scan_id_dd = gr.Dropdown(
#             choices=AVAILABLE_SCANS,
#             value=AVAILABLE_SCANS[0],
#             label="Scene (scan_id)",
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

#     # Details state + toggle button
#     details_open = gr.State(False)
#     toggle_btn = gr.Button("Show visualization details")

#     # "Dropdown-like" details panel
#     with gr.Accordion("Visualization details", open=False) as details_panel:
#         with gr.Row():
#             color_mode = gr.Dropdown(
#                 choices=["RGB", "Instance"],  # extend later if you populate segment labels
#                 value="RGB",
#                 label="Color mode",
#             )
#             point_size = gr.Slider(1, 10, value=2, step=1, label="Point size")

#         with gr.Row():
#             show_boxes = gr.Checkbox(value=False, label="Show instance bounding boxes")
#             show_axis = gr.Checkbox(value=False, label="Show world axis")
#             show_arrow = gr.Checkbox(value=True, label="Show situation arrow")
#             axis_len = gr.Slider(0.5, 5.0, value=1.5, step=0.1, label="Axis/arrow scale")

#         with gr.Row():
#             max_points = gr.Slider(10_000, 500_000, value=200_000, step=10_000, label="Max points (downsample for speed)")
#             max_boxes = gr.Slider(10, 500, value=200, step=10, label="Max boxes (cap for speed)")

#     # Main content: Viewer (left) + Chat panel (right)
#     with gr.Row():
#         with gr.Column(scale=7):
#             btn = gr.Button("Render")
#             plot = gr.Plot(elem_id="scene-plot", scale=5)
#             # info = gr.Textbox(label="Info", lines=7)

#         with gr.Column(scale=3):
#             gr.Markdown("### Ask MSR3D(PTv3 Backbone) about the scene:")
#             gr.Markdown("                                             ")
#             chat = gr.Chatbot(label="Dialogue", height=400)

#             user_msg = gr.Textbox(
#                 label="Ask a question",
#                 placeholder="Ask about the scene, situation, objects, relations, etc.",
#                 lines=3,
#             )

#             with gr.Row():
#                 send = gr.Button("Send")
#                 clear = gr.Button("Clear")

#             # context_box = gr.Textbox(label="Current context (debug)", lines=7, interactive=False)

#     # Initialize QA dropdown
#     demo.load(fn=on_scan_or_split_change, inputs=[scan_id_dd, split_filter], outputs=[qa_dd])

#     # Update QA dropdown when scan or split changes
#     scan_id_dd.change(fn=on_scan_or_split_change, inputs=[scan_id_dd, split_filter], outputs=[qa_dd])
#     split_filter.change(fn=on_scan_or_split_change, inputs=[scan_id_dd, split_filter], outputs=[qa_dd])

#     # Keep context debug updated when QA selection changes
#     # qa_dd.change(fn=update_context_box, inputs=[qa_dd], outputs=[context_box])

#     # Toggle details panel + update button label
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
#         inputs=[qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_points, max_boxes],
#         # outputs=[plot, info],
#         outputs=[plot],
#     )

#     # Chat send / submit
#     send.click(fn=chat_step, inputs=[user_msg, chat, qa_dd], outputs=[user_msg, chat])
#     user_msg.submit(fn=chat_step, inputs=[user_msg, chat, qa_dd], outputs=[user_msg, chat])
#     clear.click(fn=clear_chat, inputs=[], outputs=[chat])

# if __name__ == "__main__":
#     demo.launch()

#!/usr/bin/env python3
"""
MSQA Multi-Dataset Scene Viewer (Gradio + Plotly) — ScanNet + ARKitScenes + RScan (folder-based)

What you asked for (implemented):
- For the RScan dataset: the scene dropdown shows the folder names (scan_id).
- When rendering an RScan scene, it loads: <RSCAN_PCD_ROOT>/<scan_id>/pcds.pth

Other datasets:
- ScanNet / ARKit: still load from <PCD_ROOT>/<scan_id>.pth

UI additions retained:
1) Visualization details hidden by default (Accordion) + Show/Hide toggle button.
2) Right-side chat stub.
"""

import os
import json
import torch
import numpy as np
import gradio as gr
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

# ======================== Config ========================

SCANNET_ROOT_DIR = "/mnt/d/Thesis/data/text_annotations/msqa/scannet"
ARKIT_ROOT_DIR   = "/mnt/d/Thesis/data/text_annotations/msqa/arkitscenes"
RSCAN_ROOT_DIR   = "/mnt/d/Thesis/data/text_annotations/msqa/rscan"

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

SCANNET_PCD_ROOT = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment"
ARKIT_PCD_ROOT   = "/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd-align/pcd-align"

# IMPORTANT: RScan is folder-based, and we will load pcds.pth inside each folder
RSCAN_PCD_ROOT   = "/mnt/d/Thesis/data/MSR3D_v2_pcds/rscan_base/3RScan-ours-align/3RScan-ours-align"
RSCAN_PCD_FILE   = "pcds.pth"  # you explicitly requested this file

ONLY_SHOW_SCANS_WITH_PTH = True

DATASET_SPECS = {
    "scannet": {"json_paths": SCANNET_JSON_PATHS, "pcd_root": SCANNET_PCD_ROOT},
    "arkit":   {"json_paths": ARKIT_JSON_PATHS,   "pcd_root": ARKIT_PCD_ROOT},
    "rscan":   {"json_paths": RSCAN_JSON_PATHS,   "pcd_root": RSCAN_PCD_ROOT},
}

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
    What you asked for:
    - RScan: load <RSCAN_PCD_ROOT>/<scan_id>/pcds.pth
    - Others: load <PCD_ROOT>/<scan_id>.pth
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

# ======================== Scene load + render ========================

def load_scene(dataset_name: str, idx: int):
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

    # Attempt labels (best-effort)
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
        "segment20": None,
        "segment200": None,
        "location": location,
        "orientation": orientation,
    }

def build_plotly_figure(scene, color_mode: str, point_size: float,
                        show_boxes: bool, show_axis: bool, show_arrow: bool,
                        axis_len: float, max_points: int, max_boxes: int):

    xyz = scene["points"].reshape(-1, 3)
    N = xyz.shape[0]

    if max_points is not None and max_points > 0 and N > max_points:
        idx = np.random.default_rng(0).choice(N, size=int(max_points), replace=False)
        xyz_vis = xyz[idx]
        rgb_vis = scene["rgb01"][idx]
        inst_vis = scene["instance_labels"][idx] if scene["instance_labels"] is not None else None
    else:
        xyz_vis = xyz
        rgb_vis = scene["rgb01"]
        inst_vis = scene["instance_labels"]

    if color_mode == "RGB":
        cols = rgb_vis
    elif color_mode == "Instance":
        cols = rgb_vis if inst_vis is None else hash_colors_for_labels(inst_vis, seed=123)
    else:
        cols = rgb_vis

    cols255 = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    color_str = [f"rgb({r},{g},{b})" for r, g, b in cols255]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xyz_vis[:, 0], y=xyz_vis[:, 1], z=xyz_vis[:, 2],
        mode="markers",
        marker=dict(size=float(point_size), color=color_str, opacity=1.0),
        showlegend=False,
    ))

    if show_boxes and scene["instance_labels"] is not None:
        for t in build_instance_bboxes_as_traces(scene["points"], scene["instance_labels"], max_boxes=int(max_boxes)):
            fig.add_trace(t)

    if show_axis:
        center = xyz_vis.mean(axis=0)
        for t in make_world_axis_traces(center, axis_len=float(axis_len)):
            fig.add_trace(t)

    if show_arrow and scene["location"] is not None and scene["orientation"] is not None:
        try:
            fig.add_trace(make_situation_arrow_trace(scene["location"], scene["orientation"], scale=float(axis_len)))
        except Exception as e:
            print(f"[warn] could not render situation arrow: {e}")

    title = f"{scene['scan_id']} | split: {scene.get('split','?')}"
    if scene.get("situation"):
        title += f" | {scene['situation']}"

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

# ======================== Gradio callbacks ========================

def on_dataset_change(dataset_name: str):
    scans = AVAILABLE_SCANS_BY_DATASET[dataset_name]
    scan_val = scans[0] if scans else None
    split_val = "all"

    qa_choices = qa_choices_for_scan(dataset_name, scan_val, split_val) if scan_val else []
    qa_val = qa_choices[0][1] if qa_choices else None

    return (
        gr.update(choices=scans, value=scan_val),     # scan_id_dd
        gr.update(value=split_val),                   # split_filter
        gr.update(choices=qa_choices, value=qa_val),  # qa_dd
    )

def on_scan_or_split_change(dataset_name: str, scan_id: str, split_filter: str):
    choices = qa_choices_for_scan(dataset_name, scan_id, split_filter)
    default_val = choices[0][1] if choices else None
    return gr.update(choices=choices, value=default_val)

def render(dataset_name, global_idx, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_points, max_boxes):
    if global_idx is None:
        raise gr.Error("No QA entry selected for this scan_id/split filter.")
    idx = int(global_idx)

    scene = load_scene(dataset_name, idx)
    fig = build_plotly_figure(
        scene=scene,
        color_mode=color_mode,
        point_size=point_size,
        show_boxes=show_boxes,
        show_axis=show_axis,
        show_arrow=show_arrow,
        axis_len=axis_len,
        max_points=int(max_points),
        max_boxes=int(max_boxes),
    )
    return fig

# ---- Details panel toggle ----

def toggle_details(is_open: bool):
    new_state = not bool(is_open)
    return gr.update(open=new_state), new_state

def update_toggle_button_label(is_open: bool):
    return gr.update(value=("Hide visualization details" if is_open else "Show visualization details"))

# ---- Chat stub ----

def answer_with_model(user_msg: str, dataset_name: str, global_idx):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return ""
    return f"(stub) dataset={dataset_name}, idx={global_idx}\n\nQuestion:\n{user_msg}"

def chat_step(user_msg, history, dataset_name, global_idx):
    history = history or []
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return "", history
    model_answer = answer_with_model(user_msg, dataset_name, global_idx)
    history.append((user_msg, model_answer))
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
        "Select **dataset → scene(folder) → QA**, optionally filter by split, then render.\n\n"
        "**RScan** loads `<scan_id>/pcds.pth` inside each scan folder."
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

    details_open = gr.State(False)
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

    # Init
    demo.load(fn=on_dataset_change, inputs=[dataset_dd], outputs=[scan_id_dd, split_filter, qa_dd])

    # Dataset changes -> update scenes + QA list
    dataset_dd.change(fn=on_dataset_change, inputs=[dataset_dd], outputs=[scan_id_dd, split_filter, qa_dd])

    # Scan/split changes -> update QA list
    scan_id_dd.change(fn=on_scan_or_split_change, inputs=[dataset_dd, scan_id_dd, split_filter], outputs=[qa_dd])
    split_filter.change(fn=on_scan_or_split_change, inputs=[dataset_dd, scan_id_dd, split_filter], outputs=[qa_dd])

    # Details toggle
    toggle_btn.click(
        fn=toggle_details,
        inputs=[details_open],
        outputs=[details_panel, details_open],
    ).then(
        fn=update_toggle_button_label,
        inputs=[details_open],
        outputs=[toggle_btn],
    )

    # Render
    btn.click(
        fn=render,
        inputs=[dataset_dd, qa_dd, color_mode, point_size, show_boxes, show_axis, show_arrow, axis_len, max_points, max_boxes],
        outputs=[plot],
    )

    # Chat
    send.click(fn=chat_step, inputs=[user_msg, chat, dataset_dd, qa_dd], outputs=[user_msg, chat])
    user_msg.submit(fn=chat_step, inputs=[user_msg, chat, dataset_dd, qa_dd], outputs=[user_msg, chat])
    clear.click(fn=clear_chat, inputs=[], outputs=[chat])

if __name__ == "__main__":
    demo.launch()
