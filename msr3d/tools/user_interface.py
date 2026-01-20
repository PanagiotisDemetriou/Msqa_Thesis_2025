import os
import json
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation as R
from collections import Counter


# =========================
#  Utilities
# =========================

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def to_float_colors(rgb_uint8: np.ndarray) -> np.ndarray:
    return np.asarray(rgb_uint8, dtype=np.float32) / 255.0

def hash_colors_for_labels(labels: np.ndarray, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)
    table = {}
    for u in uniq:
        col = rng.random(3) * 0.8 + 0.2
        table[int(u)] = col.astype(np.float32)
    return np.array([table[int(x)] for x in labels], dtype=np.float32)

def print_label_stats(name: str, labels: np.ndarray, topk: int = 10) -> str:
    c = Counter(labels.tolist())
    top = c.most_common(topk)
    more = max(0, len(c) - topk)
    lines = [f"[stats] {name}: {len(c)} unique labels; top {topk} by frequency:"]
    for k, v in top:
        lines.append(f"   ID {k:>4}: {v}")
    if more:
        lines.append(f"   ... and {more} more")
    return "\n".join(lines)

def quat_to_forward_vector(quat_xyzw: np.ndarray, forward_axis="z") -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    rot = R.from_quat(quat_xyzw).as_matrix()
    if forward_axis.lower() == "z":
        local_fwd = np.array([0.0, 0.0, 1.0])
    else:
        local_fwd = np.array([1.0, 0.0, 0.0])
    fwd = rot @ local_fwd
    n = np.linalg.norm(fwd) + 1e-12
    return (fwd / n).astype(np.float32)

def _parse_vec3(v) -> np.ndarray:
    # Accept dict {"x","y","z"} or {"_x","_y","_z"} or list-like [x,y,z]
    if isinstance(v, dict):
        if all(k in v for k in ("x", "y", "z")):
            return np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        if all(k in v for k in ("_x", "_y", "_z")):
            return np.array([v["_x"], v["_y"], v["_z"]], dtype=np.float32)
        # try nesting
        for key in ("pos", "position", "location", "translation"):
            if key in v:
                return _parse_vec3(v[key])
        raise ValueError(f"Unsupported location dict keys: {list(v.keys())}")
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"location must have 3 values, got: {arr}")
    return arr[:3]

def _parse_quat_xyzw(q) -> np.ndarray:
    # Accept dict {"_x","_y","_z","_w"} or {"x","y","z","w"} or list-like [x,y,z,w]
    if isinstance(q, dict):
        if all(k in q for k in ("_x", "_y", "_z", "_w")):
            return np.array([q["_x"], q["_y"], q["_z"], q["_w"]], dtype=np.float32)
        if all(k in q for k in ("x", "y", "z", "w")):
            return np.array([q["x"], q["y"], q["z"], q["w"]], dtype=np.float32)
        # try nesting
        for key in ("ori", "orientation", "rotation", "quat", "quaternion"):
            if key in q:
                return _parse_quat_xyzw(q[key])
        raise ValueError(f"Unsupported orientation dict keys: {list(q.keys())}")
    arr = np.asarray(q, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"orientation must have 4 values (x,y,z,w), got: {arr}")
    return arr[:4]


# =========================
#  Geometry helpers
# =========================

def make_thin_axis(size=0.5, thickness=0.02):
    cyl_h = size * 0.85
    cone_h = size - cyl_h
    rad = max(1e-6, size * thickness)
    cone_rad = rad * 1.8

    def arrow(color, R_mat=np.eye(3)):
        a = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=rad,
            cone_radius=cone_rad,
            cylinder_height=cyl_h,
            cone_height=cone_h,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        a.compute_vertex_normals()
        a.paint_uniform_color(color)
        a.rotate(R_mat, center=(0, 0, 0))
        return a

    Rx = o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.pi/2, 0.0))
    Ry = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi/2, 0.0, 0.0))
    Rz = np.eye(3)
    return [
        arrow([1, 0, 0], Rx),
        arrow([0, 1, 0], Ry),
        arrow([0, 0, 1], Rz),
    ]

def create_situation_arrow(origin: np.ndarray, direction: np.ndarray, scale=0.5):
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    dn = np.linalg.norm(direction) + 1e-12
    direction = direction / dn

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02,
        cone_radius=0.06,
        cylinder_height=scale * 0.8,
        cone_height=scale * 0.2
    )
    arrow.compute_vertex_normals()

    default_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # Open3D arrow points +Z
    if not np.allclose(direction, default_dir):
        rot, _ = R.align_vectors([direction], [default_dir])
        arrow.rotate(rot.as_matrix(), center=(0, 0, 0))

    arrow.translate(origin)
    arrow.paint_uniform_color([1.0, 0.0, 0.0])
    return arrow

def build_instance_bbox_linesets(xyz: np.ndarray, instance: Optional[np.ndarray]) -> Tuple[List[o3d.geometry.LineSet], List[int]]:
    if instance is None:
        return [], []

    xyz = np.asarray(xyz)
    inst = np.asarray(instance).reshape(-1)
    uniq = np.unique(inst)

    cols_for_uniq = hash_colors_for_labels(uniq, seed=123)
    col_tab = {int(u): cols_for_uniq[i].tolist() for i, u in enumerate(uniq)}

    linesets, ids = [], []
    for u in uniq:
        mask = (inst == u)
        if not np.any(mask):
            continue
        pts = xyz[mask]
        if pts.shape[0] < 2:
            continue

        aabb = o3d.geometry.AxisAlignedBoundingBox(pts.min(0), pts.max(0))
        ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)

        color = np.array(col_tab[int(u)], dtype=np.float32)[None, :]
        ls.colors = o3d.utility.Vector3dVector(np.repeat(color, len(ls.lines), axis=0))

        linesets.append(ls)
        ids.append(int(u))

    return linesets, ids


# =========================
#  App state
# =========================

@dataclass
class AppState:
    color_mode: str = "RGB"
    show_bboxes: bool = False
    show_axis: bool = False
    show_arrow: bool = True
    point_size: float = 2.0


# =========================
#  Main app
# =========================

class SceneChatApp019:
    def __init__(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        instance_labels: Optional[np.ndarray] = None,
        location: Optional[object] = None,
        orientation: Optional[object] = None,
        situation: Optional[str] = None,
        axis_size: float = 3.5,
        axis_thickness: float = 0.0075,
        pt_size_init: float = 2.0,
    ):
        self.state = AppState(point_size=float(pt_size_init))
        self.situation = situation or "Scene Viewer"

        # ---- Prepare point cloud ----
        self.xyz = np.asarray(points).reshape(-1, 3).astype(np.float32)

        # input colors are in [-1, 1], convert to [0, 1]
        c = np.asarray(colors).reshape(-1, 3).astype(np.float32)
        self.rgb = np.clip((c + 1.0) / 2.0, 0.0, 1.0)

        self.instance_labels = None if instance_labels is None else np.asarray(instance_labels).reshape(-1)

        # Color modes (0.19-safe: keep it simple; add more later)
        self.color_modes: Dict[str, np.ndarray] = {"RGB": self.rgb}
        if self.instance_labels is not None:
            self.color_modes["Instance"] = hash_colors_for_labels(self.instance_labels, seed=123)

        # Overlays
        self.axis_geoms = make_thin_axis(size=axis_size, thickness=axis_thickness)
        self.bbox_linesets, self.bbox_ids = build_instance_bbox_linesets(self.xyz, self.instance_labels)

        self.arrow_mesh = None
        self.state.show_arrow = False
        if location is not None and orientation is not None:
            try:
                origin = _parse_vec3(location)
                quat = _parse_quat_xyzw(orientation)
                direction = quat_to_forward_vector(quat, forward_axis="z")  # change to "x" if needed
                self.arrow_mesh = create_situation_arrow(origin, direction, scale=0.5)
                self.state.show_arrow = True
            except Exception as e:
                print(f"[warn] Could not create situation arrow. Disabling. Reason: {e}")
                self.arrow_mesh = None
                self.state.show_arrow = False

        # GUI
        self._init_gui()
        self._build_scene()

    # ---------- GUI ----------

    def _init_gui(self):
        self.app = gui.Application.instance
        self.app.initialize()

        self.window = self.app.create_window(self.situation, 1400, 800)

        em = self.window.theme.font_size
        margin = int(0.5 * em)

        self.root = gui.Horiz(0, gui.Margins(margin, margin, margin, margin))
        self.window.add_child(self.root)

        # Left: Scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.root.add_child(self.scene_widget)

        # Right: panel
        self.panel = gui.Vert(0, gui.Margins(margin, margin, margin, margin))
        self.panel.preferred_width = 420
        self.root.add_child(self.panel)

        # ---- Controls ----
        self.panel.add_child(gui.Label("Visualization"))

        self.panel.add_child(gui.Label("Color mode"))
        self.mode_combo = gui.Combobox()
        for name in self.color_modes.keys():
            self.mode_combo.add_item(name)
        self.mode_combo.selected_text = self.state.color_mode
        self.mode_combo.set_on_selection_changed(self._on_color_mode)
        self.panel.add_child(self.mode_combo)

        self.chk_bboxes = gui.Checkbox("Instance bounding boxes")
        self.chk_bboxes.checked = self.state.show_bboxes
        self.chk_bboxes.set_on_checked(self._on_bboxes)
        self.panel.add_child(self.chk_bboxes)

        self.chk_axis = gui.Checkbox("World axis")
        self.chk_axis.checked = self.state.show_axis
        self.chk_axis.set_on_checked(self._on_axis)
        self.panel.add_child(self.chk_axis)

        self.chk_arrow = gui.Checkbox("Situation arrow")
        self.chk_arrow.enabled = (self.arrow_mesh is not None)
        self.chk_arrow.checked = self.state.show_arrow
        self.chk_arrow.set_on_checked(self._on_arrow)
        self.panel.add_child(self.chk_arrow)

        self.panel.add_child(gui.Label("Point size"))
        self.pt_slider = gui.Slider(gui.Slider.DOUBLE)
        self.pt_slider.set_limits(1.0, 12.0)
        self.pt_slider.double_value = float(self.state.point_size)
        self.pt_slider.set_on_value_changed(self._on_point_size)
        self.panel.add_child(self.pt_slider)

        self.btn_stats = gui.Button("Print instance stats")
        self.btn_stats.enabled = (self.instance_labels is not None)
        self.btn_stats.set_on_clicked(self._on_stats)
        self.panel.add_child(self.btn_stats)

        self.panel.add_fixed(margin)

        # ---- Chat ----
        self.panel.add_child(gui.Label("Chat"))

        # Transcript: use Label (stable in 0.19.0)
        self.chat_transcript = gui.Label("Model chat ready.\n")
        if hasattr(self.chat_transcript, "text_wrapping"):
            self.chat_transcript.text_wrapping = gui.Label.WrapMode.WORD
        self.panel.add_child(self.chat_transcript)

        # Input: TextEdit (0.19 uses set_text/get_text)
        self.chat_input = gui.TextEdit()
        if hasattr(self.chat_input, "set_text"):
            self.chat_input.set_text("")
        if hasattr(self.chat_input, "preferred_height"):
            self.chat_input.preferred_height = int(em * 3.5)
        self.panel.add_child(self.chat_input)

        row = gui.Horiz(0, gui.Margins(0, 0, 0, 0))
        self.btn_send = gui.Button("Send")
        self.btn_send.set_on_clicked(self._on_send)
        self.btn_clear = gui.Button("Clear")
        self.btn_clear.set_on_clicked(self._on_clear)
        row.add_child(self.btn_send)
        row.add_fixed(int(0.5 * em))
        row.add_child(self.btn_clear)
        self.panel.add_child(row)

        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(lambda: True)

    def _on_layout(self, ctx):
        r = self.window.content_rect
        self.root.frame = r

        panel_w = self.panel.preferred_width
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)
        self.panel.frame = gui.Rect(r.x + (r.width - panel_w), r.y, panel_w, r.height)

    # ---------- Scene ----------

    def _pcd_geometry(self, colors: np.ndarray) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
        return pcd

    def _build_scene(self):
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        scene = self.scene_widget.scene

        # Materials
        self.pcd_mat = rendering.MaterialRecord()
        self.pcd_mat.shader = "defaultUnlit"
        self.pcd_mat.point_size = float(self.state.point_size)

        self.lines_mat = rendering.MaterialRecord()
        self.lines_mat.shader = "unlitLine"
        self.lines_mat.line_width = 2.0

        self.mesh_mat = rendering.MaterialRecord()
        self.mesh_mat.shader = "defaultLit"

        # Add point cloud
        cols = self.color_modes.get(self.state.color_mode, self.rgb)
        self.pcd = self._pcd_geometry(cols)
        scene.scene.add_geometry("pcd", self.pcd, self.pcd_mat)

        # Overlays
        if self.state.show_axis:
            for i, g in enumerate(self.axis_geoms):
                scene.scene.add_geometry(f"axis_{i}", g, self.mesh_mat)

        if self.state.show_arrow and self.arrow_mesh is not None:
            scene.scene.add_geometry("arrow", self.arrow_mesh, self.mesh_mat)

        if self.state.show_bboxes and len(self.bbox_linesets) > 0:
            for i, ls in enumerate(self.bbox_linesets):
                scene.scene.add_geometry(f"bbox_{i}", ls, self.lines_mat)

        # Camera fit
        bbox = self.pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        self.scene_widget.setup_camera(60.0, bbox, center)



    def _update_pcd(self):
        scene = self.scene_widget.scene
        if scene.scene.has_geometry("pcd"):
            scene.scene.remove_geometry("pcd")
        cols = self.color_modes.get(self.state.color_mode, self.rgb)
        self.pcd = self._pcd_geometry(cols)
        self.pcd_mat.point_size = float(self.state.point_size)
        scene.scene.add_geometry("pcd", self.pcd, self.pcd_mat)

    def _set_overlay(self, prefix: str, visible: bool, geoms: List[o3d.geometry.Geometry], mat: rendering.MaterialRecord):
        scene = self.scene_widget.scene
        if visible:
            for i, g in enumerate(geoms):
                name = f"{prefix}_{i}"
                if not scene.scene.has_geometry(name):
                    scene.scene.add_geometry(name, g, mat)
        else:
            for i in range(len(geoms)):
                name = f"{prefix}_{i}"
                if scene.scene.has_geometry(name):
                    scene.scene.remove_geometry(name)

    # ---------- UI callbacks ----------

    def _append_chat(self, line: str):
        # Label is stable: just rebuild text
        self.chat_transcript.text = self.chat_transcript.text + line + "\n"

    def _on_color_mode(self, text, idx):
        self.state.color_mode = text
        self._update_pcd()
        self._append_chat(f"[viewer] Color mode: {text}")

    def _on_bboxes(self, checked: bool):
        if checked and len(self.bbox_linesets) == 0:
            self._append_chat("[viewer] No instance boxes available.")
            self.chk_bboxes.checked = False
            self.state.show_bboxes = False
            return
        self.state.show_bboxes = bool(checked)
        self._set_overlay("bbox", self.state.show_bboxes, self.bbox_linesets, self.lines_mat)
        self._append_chat(f"[viewer] Boxes: {'ON' if self.state.show_bboxes else 'OFF'}")

    def _on_axis(self, checked: bool):
        self.state.show_axis = bool(checked)
        self._set_overlay("axis", self.state.show_axis, self.axis_geoms, self.mesh_mat)
        self._append_chat(f"[viewer] Axis: {'ON' if self.state.show_axis else 'OFF'}")

    def _on_arrow(self, checked: bool):
        self.state.show_arrow = bool(checked)
        scene = self.scene_widget.scene
        if self.arrow_mesh is None:
            self._append_chat("[viewer] No arrow available.")
            self.chk_arrow.checked = False
            self.state.show_arrow = False
            return

        if self.state.show_arrow:
            if not scene.scene.has_geometry("arrow"):
                scene.scene.add_geometry("arrow", self.arrow_mesh, self.mesh_mat)
        else:
            if scene.scene.has_geometry("arrow"):
                scene.scene.remove_geometry("arrow")

        self._append_chat(f"[viewer] Arrow: {'ON' if self.state.show_arrow else 'OFF'}")

    def _on_point_size(self, val: float):
        self.state.point_size = float(val)
        self._update_pcd()

    def _on_stats(self):
        if self.instance_labels is None:
            self._append_chat("[stats] No instance labels available.")
            return
        self._append_chat(print_label_stats("Instances", self.instance_labels))

    # ---------- Chat ----------

    def _get_input_text(self) -> str:
        if hasattr(self.chat_input, "get_text"):
            return self.chat_input.get_text()
        # fallback (should not happen in 0.19)
        return ""

    def _set_input_text(self, s: str):
        if hasattr(self.chat_input, "set_text"):
            self.chat_input.set_text(s)

    def _on_clear(self):
        self.chat_transcript.text = "Model chat ready.\n"

    def _on_send(self):
        msg = (self._get_input_text() or "").strip()
        if not msg:
            return
        self._set_input_text("")
        self._append_chat(f"User: {msg}")
        self.btn_send.enabled = False

        t = threading.Thread(target=self._chat_worker, args=(msg,), daemon=True)
        t.start()

    def _chat_worker(self, user_text: str):
        try:
            response = self._run_model_inference(user_text)
        except Exception as e:
            response = f"[error] {e}"

        def _update():
            self._append_chat(f"Model: {response}")
            self.btn_send.enabled = True

        gui.Application.instance.post_to_main_thread(self.window, _update)

    def _run_model_inference(self, user_text: str) -> str:
        # Replace with your real inference call
        time.sleep(0.3)
        return f"(stub) Received: '{user_text}'. Replace _run_model_inference()."

    # ---------- Run ----------

    def run(self):
        self.app.run()


# =========================
#  Entry point
# =========================

if __name__ == "__main__":
    # Your paths (keep as you have them)
    root_dir = "/mnt/d/Thesis/data/text_annotations/msqa/scannet"
    pcd_root = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/"

    data_dict = load_json(f"{root_dir}/msqa_scannet_test.json")

    data_id = 0
    qa_pair = data_dict[data_id]
    scan_id = qa_pair["scan_id"]
    pcd_path = os.path.join(pcd_root, scan_id + ".pth")

    pcd_data = torch.load(pcd_path, weights_only=False)
    points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]

    # Your preprocessing: colors -> [-1, 1]
    colors = colors / 127.5 - 1.0

    print(f"[info] Loading scene: {scan_id}")
    print(f"[info] Situation: {qa_pair.get('situation', '')}")

    app = SceneChatApp019(
        points=points,
        colors=colors,
        instance_labels=instance_labels,
        location=qa_pair.get("location", None),
        orientation=qa_pair.get("orientation", None),
        situation=qa_pair.get("situation", None),
        pt_size_init=2.0,
        axis_size=3.5,
        axis_thickness=0.0075,
    )
    app.run()
