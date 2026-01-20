import os
import torch
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from collections import Counter

# ======================== Utils ========================

def quaternion_to_euler(quaternion):
    euler_angles = R.from_quat(quaternion).as_euler('xyz', degrees=False)
    x_angle, y_angle, z_angle = euler_angles
    return [x_angle, y_angle, z_angle]

def get_view_vector(quaternion):
    angle = quaternion_to_euler(quaternion)[-1]
    view_vector = [np.cos(angle), np.sin(angle), 0.0]
    return view_vector

def load_json(path):
    with open(path, 'r') as f:
        json_file = json.load(f)
    return json_file

def normalize_colors(c):
    c = c.astype(np.float32)
    if c.max() > 1.5:
        c = c / 255.0
    return np.clip(c, 0.0, 1.0)

def to_float_colors(rgb_uint8):
    return np.asarray(rgb_uint8, dtype=np.float32) / 255.0

# Official ScanNet20 palette
SCANNET20_COLORS = to_float_colors(np.array([
    [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120],
    [188, 189, 34], [140, 86, 75], [255, 152, 150], [214, 39, 40],
    [197, 176, 213], [148, 103, 189], [196, 156, 148], [23, 190, 207],
    [247, 182, 210], [219, 219, 141], [255, 127, 14], [158, 218, 229],
    [44, 160, 44], [112, 128, 144], [227, 119, 194], [82, 84, 163],
], dtype=np.uint8))

def hash_colors_for_labels(labels, seed=0):
    """Deterministic random colors for arbitrary label IDs."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)
    table = {}
    for u in uniq:
        col = rng.random(3) * 0.8 + 0.2
        table[int(u)] = col.astype(np.float32)
    return np.array([table[int(x)] for x in labels], dtype=np.float32)

def palette_colors_for_labels(labels, palette):
    """Map labels onto a fixed palette (wrap-around)."""
    uniq = np.unique(labels)
    table = {}
    K = len(palette)
    for i, u in enumerate(uniq):
        table[int(u)] = palette[i % K]
    return np.array([table[int(x)] for x in labels], dtype=np.float32)

def print_label_stats(name, labels):
    c = Counter(labels.tolist())
    top = c.most_common(10)
    more = max(0, len(c) - 10)
    print(f"[stats] {name}: {len(c)} unique labels; top 10 by frequency:")
    for k, v in top:
        print(f"   ID {k:>4}: {v}")
    if more:
        print(f"   ... and {more} more")

# ======================== Geometry Utils ========================

def build_instance_bboxes(xyz, instance, color_from_instance=True):
    if instance is None:
        return [], []

    xyz = np.asarray(xyz)
    inst = np.asarray(instance).reshape(-1)
    uniq = np.unique(inst)

    col_tab = {}
    if color_from_instance:
        cols = hash_colors_for_labels(uniq, seed=123)
        for u, c in zip(uniq, cols):
            col_tab[int(u)] = c

    bboxes, ids = [], []
    for u in uniq:
        mask = (inst == u)
        if not np.any(mask):
            continue
        pts = xyz[mask]
        if pts.shape[0] < 2:
            continue
        bb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=pts.min(0), max_bound=pts.max(0)
        )
        if color_from_instance:
            bb.color = col_tab[int(u)].tolist()
        else:
            bb.color = [0.9, 0.9, 0.9]
        bboxes.append(bb)
        ids.append(int(u))
    return bboxes, ids

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
        a.paint_uniform_color(color)
        a.rotate(R_mat, center=(0, 0, 0))
        return a

    Rx = o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.pi/2, 0.0))
    Ry = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi/2, 0.0, 0.0))
    Rz = np.eye(3)

    return [arrow([1, 0, 0], Rx), arrow([0, 1, 0], Ry), arrow([0, 0, 1], Rz)]

def create_situation_arrow(origin, direction, scale=0.5):
    """Creates an arrow for situation visualization."""
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction) * scale

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02,
        cone_radius=0.06,
        cylinder_height=scale * 0.8,
        cone_height=scale * 0.2
    )

    default_dir = np.array([0, 0, 1])
    if not np.allclose(direction, default_dir):
        rotation_matrix = R.align_vectors([direction], [default_dir])[0].as_matrix()
        arrow.rotate(rotation_matrix)

    arrow.translate(origin)
    arrow.paint_uniform_color([1, 0, 0])

    return arrow

def align_situation(pos, ori, scene_center, align_matrix):
    """Transform location and orientation to align with pcd."""
    if isinstance(pos, dict):
        pos = [pos['x'], pos['y'], pos['z']]
    pos = np.array(pos)

    if isinstance(ori, dict):
        ori = [ori['_x'], ori['_y'], ori['_z'], ori['_w']]
    ori = np.array(ori)

    pos_new = pos.reshape(1, 3) @ align_matrix.T
    pos_new += scene_center
    pos_new = pos_new.reshape(-1)

    ori = R.from_quat(ori).as_matrix()
    ori_new = align_matrix @ ori
    flip_matrix = R.from_euler('z', 180, degrees=True).as_matrix()
    ori_new = flip_matrix @ ori_new
    ori_new = R.from_matrix(ori_new).as_quat()
    ori_new = ori_new.reshape(-1)
    return pos_new, ori_new

# ======================== Main Visualization ========================

def visualize_scene_interactive(points, colors, instance_labels, location=None, 
                                orientation=None, situation=None, segment20=None, 
                                segment200=None, pt_size_init=2.0, axis_size=0.5, 
                                axis_thickness=0.02):
    """Interactive visualization with multiple color modes and situation arrow."""
    
    # Prepare point cloud
    xyz = np.asarray(points).reshape(-1, 3).astype(np.float32)
    # Colors are already in [-1, 1] range from the loader, convert to [0, 1]
    colors = np.asarray(colors).reshape(-1, 3).astype(np.float32)
    colors = (colors + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals for better shading
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.normalize_normals()
    
    # Prepare instance bboxes
    boxes, box_ids = build_instance_bboxes(xyz, instance_labels)
    if boxes:
        print(f"[info] Prepared {len(boxes)} instance bounding boxes.")
    
    # Create situation arrow
    situation_arrow = None
    if location is not None and orientation is not None:
        situation_arrow = create_situation_arrow(location, orientation, scale=0.5)
        print(f"[info] Situation arrow created at {location}")
    
    # Prepare color modes
    color_modes = []
    base_rgb = np.asarray(pcd.colors).copy()
    color_modes.append(("RGB", base_rgb))
    
    if instance_labels is not None:
        inst_cols = hash_colors_for_labels(instance_labels, seed=123)
        color_modes.append(("Instance", inst_cols))
    
    if segment20 is not None:
        seg20_cols = palette_colors_for_labels(segment20, SCANNET20_COLORS)
        color_modes.append(("Segment20", seg20_cols))
    
    if segment200 is not None:
        seg200_cols = hash_colors_for_labels(segment200, seed=200)
        color_modes.append(("Segment200", seg200_cols))
    
    # Print controls
    print("\n[controls]")
    print("  1 → RGB colors")
    print("  2 → Instance segmentation")
    print("  3 → Segment20 (if available)")
    print("  4 → Segment200 (if available)")
    print("  c → Cycle through color modes")
    print("  b → Toggle instance bounding boxes")
    print("  a → Toggle situation arrow")
    print("  x → Toggle world axis")
    print("  + / = → Increase point size")
    print("  - / _ → Decrease point size")
    print("  h → Print help and stats")
    print("  q / ESC → Close window\n")
    
    # Pre-create axis geometries once
    axis_geoms = make_thin_axis(size=axis_size, thickness=axis_thickness)
    
    # State
    state = {
        "mode_idx": 0,
        "boxes_on": False,
        "arrow_on": True if situation_arrow else False,
        "axis_on": False,
        "pt_size": float(pt_size_init),
        "axis_geoms": axis_geoms,
    }
    
    # Callbacks
    def set_mode(vis, idx):
        idx = max(0, min(idx, len(color_modes) - 1))
        state["mode_idx"] = idx
        name, cols = color_modes[idx]
        pcd.colors = o3d.utility.Vector3dVector(cols)
        vis.update_geometry(pcd)
        vis.update_renderer()
        print(f"[mode] -> {name}")
        return False
    
    def cb_1(vis):
        for i, (name, _) in enumerate(color_modes):
            if name.lower() == "rgb":
                return set_mode(vis, i)
        return False
    
    def cb_2(vis):
        for i, (name, _) in enumerate(color_modes):
            if name.lower() == "instance":
                return set_mode(vis, i)
        print("[info] Instance labels not available.")
        return False
    
    def cb_3(vis):
        for i, (name, _) in enumerate(color_modes):
            if name.lower() == "segment20":
                return set_mode(vis, i)
        print("[info] Segment20 labels not available.")
        return False
    
    def cb_4(vis):
        for i, (name, _) in enumerate(color_modes):
            if name.lower() == "segment200":
                return set_mode(vis, i)
        print("[info] Segment200 labels not available.")
        return False
    
    def cb_c(vis):
        return set_mode(vis, (state["mode_idx"] + 1) % len(color_modes))
    
    def cb_b(vis):
        if not boxes:
            print("[info] No instance boxes available.")
            return False
        if not state["boxes_on"]:
            for bb in boxes:
                vis.add_geometry(bb)
            state["boxes_on"] = True
            print(f"[mode] Instance boxes: ON ({len(boxes)})")
        else:
            for bb in boxes:
                vis.remove_geometry(bb, reset_bounding_box=False)
            state["boxes_on"] = False
            print("[mode] Instance boxes: OFF")
        vis.update_renderer()
        return False
    
    def cb_a(vis):
        if situation_arrow is None:
            print("[info] No situation arrow available.")
            return False
        if not state["arrow_on"]:
            vis.add_geometry(situation_arrow)
            state["arrow_on"] = True
            print("[mode] Situation arrow: ON")
        else:
            vis.remove_geometry(situation_arrow, reset_bounding_box=False)
            state["arrow_on"] = False
            print("[mode] Situation arrow: OFF")
        vis.update_renderer()
        return False
    
    def cb_x(vis):
        
        if not state["axis_on"]:
            for g in axis_geoms:
                vis.add_geometry(g)
            state["axis_on"] = True
            print("[mode] Axis: ON")
        else:
            for g in axis_geoms:
                vis.remove_geometry(g, reset_bounding_box=False)
            state["axis_on"] = False
            print("[mode] Axis: OFF")
        vis.update_renderer()
        return False
    
    def _apply_point_size(vis):
        ro = vis.get_render_option()
        ro.point_size = float(max(1.0, min(state["pt_size"], 50.0)))
        vis.update_renderer()
    
    def cb_plus(vis):
        state["pt_size"] *= 1.25
        _apply_point_size(vis)
        print(f"[mode] Point size: {state['pt_size']:.2f}")
        return False
    
    def cb_minus(vis):
        state["pt_size"] /= 1.25
        _apply_point_size(vis)
        print(f"[mode] Point size: {state['pt_size']:.2f}")
        return False
    
    def cb_h(vis):
        print("\n[controls]")
        print("  1 → RGB colors")
        print("  2 → Instance segmentation")
        print("  3 → Segment20 (if available)")
        print("  4 → Segment200 (if available)")
        print("  c → Cycle through color modes")
        print("  b → Toggle instance bounding boxes")
        print("  a → Toggle situation arrow")
        print("  x → Toggle world axis")
        print("  + / = → Increase point size")
        print("  - / _ → Decrease point size")
        print("  h → Print this help")
        print("  q / ESC → Close window\n")
        if instance_labels is not None:
            print_label_stats("Instances", instance_labels)
        return False
    
    # Key mapping
    key_to_callback = {
        ord("1"): cb_1, ord("2"): cb_2, ord("3"): cb_3, ord("4"): cb_4,
        ord("c"): cb_c, ord("C"): cb_c,
        ord("h"): cb_h, ord("H"): cb_h,
        ord("b"): cb_b, ord("B"): cb_b,
        ord("a"): cb_a, ord("A"): cb_a,
        ord("x"): cb_x, ord("X"): cb_x,
        ord("+"): cb_plus, ord("="): cb_plus,
        ord("-"): cb_minus, ord("_"): cb_minus,
    }
    
    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    window_title = f"Scene Viewer - {situation}" if situation else "Scene Viewer"
    vis.create_window(window_name=window_title)
    vis.add_geometry(pcd)
    
    # Add situation arrow by default
    if situation_arrow and state["arrow_on"]:
        vis.add_geometry(situation_arrow)
    
    # Register callbacks
    for k, cb in key_to_callback.items():
        vis.register_key_callback(k, cb)
    
    # Set initial render options
    ro = vis.get_render_option()
    ro.point_size = float(pt_size_init)
    
    vis.run()
    vis.destroy_window()

# ======================== Main ========================

if __name__ == "__main__":
    # Load MSQA data
    #root_dir = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/text_annotations/scannet/scannet"
    root_dir = "/mnt/d/Thesis/data/text_annotations/msqa/scannet"
    data_dict = load_json(f"{root_dir}/msqa_scannet_test.json")
    #pcd_root = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/"
    pcd_root = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/"
    # Process each QA pair
    for data_id in range(1):
        qa_pair = data_dict[data_id]
        scan_id = qa_pair['scan_id']
        pcd_path = os.path.join(pcd_root, scan_id + ".pth")
        
        pcd_data = torch.load(pcd_path, weights_only=False)
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
        colors = colors / 127.5 - 1
        
        print(f"\n[info] Loading scene {data_id + 1}/10: {scan_id}")
        print(f"[info] Situation: {qa_pair['situation']}")
        
        visualize_scene_interactive(
            points=points,
            colors=colors,
            instance_labels=instance_labels,
            location=qa_pair['location'],
            orientation=qa_pair['orientation'],
            situation=qa_pair['situation'],
            pt_size_init=2.0,
            axis_size=3.5,
            axis_thickness=0.0075,
        )
