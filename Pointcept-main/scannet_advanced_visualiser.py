# view_scannet_npy.py
import argparse, os, sys
import numpy as np
import open3d as o3d
from collections import Counter

# -------------------------- Utils --------------------------

def load_array(path):
    return np.load(path)

def normalize_colors(c):
    c = c.astype(np.float32)
    if c.max() > 1.5:  # likely 0-255
        c = c / 255.0
    return np.clip(c, 0.0, 1.0)

def to_float_colors(rgb_uint8):
    return np.asarray(rgb_uint8, dtype=np.float32) / 255.0

# Official-ish ScanNet20 palette (RGB 0-255, converted to 0-1)
SCANNET20_COLORS = to_float_colors(np.array([
    [174, 199, 232],
    [152, 223, 138],
    [31, 119, 180],
    [255, 187, 120],
    [188, 189, 34],
    [140, 86, 75],
    [255, 152, 150],
    [214, 39, 40],
    [197, 176, 213],
    [148, 103, 189],
    [196, 156, 148],
    [23, 190, 207],
    [247, 182, 210],
    [219, 219, 141],
    [255, 127, 14],
    [158, 218, 229],
    [44, 160, 44],
    [112, 128, 144],
    [227, 119, 194],
    [82, 84, 163],
], dtype=np.uint8))

def hash_colors_for_labels(labels, seed=0):
    """Deterministic 'random' colors for arbitrary label IDs."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)
    table = {}
    for u in uniq:
        # brighter palette (avoid too dark)
        col = rng.random(3) * 0.8 + 0.2
        table[int(u)] = col.astype(np.float32)
    return np.array([table[int(x)] for x in labels], dtype=np.float32)

def palette_colors_for_labels(labels, palette):
    """Map arbitrary integer labels onto a fixed palette (wrap-around)."""
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

# -------------------- Downsample (label-safe) --------------------

# def downsample_with_label_propagation(pcd, orig_xyz, orig_colors, lbl_dict, voxel_size, fast=False):
#     print(f"[info] Voxel downsampling at {voxel_size} m")
#     pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

#     if fast or (lbl_dict is None and orig_colors is None):
#         return pcd_down, None  # no label cache

#     # Build KDTree on original points to pull attributes onto down-sampled points
#     print("[info] Relabeling downsampled points via nearest neighbor...")
#     kdtree = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(orig_xyz))
#     down_pts = np.asarray(pcd_down.points)

#     # Prepare caches
#     cache = {"rgb": None, "instance": None, "segment20": None, "segment200": None}

#     if orig_colors is not None:
#         rgb = np.zeros((len(down_pts), 3), dtype=np.float32)
#     else:
#         rgb = None

#     lbl_arrays = {}
#     if lbl_dict is not None:
#         for k, arr in lbl_dict.items():
#             if arr is not None:
#                 lbl_arrays[k] = np.zeros(len(down_pts), dtype=arr.dtype)

#     for i, q in enumerate(down_pts):
#         _, idxs, _ = kdtree.search_knn_vector_3d(q, 1)
#         j = idxs[0]
#         if rgb is not None:
#             rgb[i] = orig_colors[j]
#         for k in lbl_arrays.keys():
#             lbl_arrays[k][i] = lbl_dict[k][j]

#     if rgb is not None:
#         cache["rgb"] = rgb
#     for k, arr in lbl_arrays.items():
#         cache[k] = arr

#     return pcd_down, cache


def build_instance_bboxes(xyz, instance, color_from_instance=True):
    if instance is None:
        return [], []

    xyz = np.asarray(xyz)
    inst = np.asarray(instance).reshape(-1)
    uniq = np.unique(inst)

    # Color table (reuse the instance hashing so box ≈ point colors)
    col_tab = {}
    if color_from_instance:
        cols = hash_colors_for_labels(uniq, seed=123)  # 0..1 floats
        for u, c in zip(uniq, cols):
            col_tab[int(u)] = c

    bboxes, ids = [], []
    for u in uniq:
        mask = (inst == u)
        if not np.any(mask):
            continue
        pts = xyz[mask]
        # Safety (skip degenerate instances)
        if pts.shape[0] < 2:
            continue
        bb = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=pts.min(0), max_bound=pts.max(0)
        )
        # Set a visible (not-too-dark) color
        if color_from_instance:
            bb.color = col_tab[int(u)].tolist()
        else:
            bb.color = [0.9, 0.9, 0.9]
        bboxes.append(bb)
        ids.append(int(u))
    return bboxes, ids

def make_thin_axis(size=0.5, thickness=0.02):

    # Arrow dimensions
    cyl_h = size * 0.85
    cone_h = size - cyl_h
    rad = max(1e-6, size * thickness)
    cone_rad = rad * 1.8

    def arrow(color, R=np.eye(3)):
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
        a.rotate(R, center=(0, 0, 0))
        return a

    
    Rx = o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.pi/2, 0.0))  
    Ry = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi/2, 0.0, 0.0))   
    Rz = np.eye(3)                                                        

    ax = arrow([1, 0, 0], Rx)  # X
    ay = arrow([0, 1, 0], Ry)  # Y
    az = arrow([0, 0, 1], Rz)  # Z
    return [ax, ay, az]

# -------------------------- Main --------------------------
def main(scene_dir, save_ply=None, estimate_normals=False,
         fast_down=False, pt_size_init=2.0, axis_on=False, axis_size=0.5, axis_thickness=0.02):

    xyz_path    = os.path.join(scene_dir, "coord.npy")
    color_path  = os.path.join(scene_dir, "color.npy")
    normal_path = os.path.join(scene_dir, "normal.npy")

    if not os.path.isfile(xyz_path):
        print(f"[error] Missing coord.npy at: {xyz_path}")
        sys.exit(1)

    xyz = load_array(xyz_path)  # (N,3)
    colors = load_array(color_path) if os.path.isfile(color_path) else None
    normals = load_array(normal_path) if os.path.isfile(normal_path) else None

    instance = segment20 = segment200 = None
    if scene_dir.startswith(("scannet/val/", "scannet/train/")):
        inst_p = os.path.join(scene_dir, "instance.npy")
        s20_p  = os.path.join(scene_dir, "segment20.npy")
        s200_p = os.path.join(scene_dir, "segment200.npy")
        if os.path.isfile(inst_p):
            instance = load_array(inst_p)
        if os.path.isfile(s20_p):
            segment20 = load_array(s20_p)
        if os.path.isfile(s200_p):
            segment200 = load_array(s200_p)

    # Ensure shapes
    xyz = np.asarray(xyz).reshape(-1, 3).astype(np.float32)
    if colors is not None:
        colors = normalize_colors(np.asarray(colors).reshape(-1, 3))
        if len(colors) != len(xyz):
            print(f"[warn] color count {len(colors)} != coord count {len(xyz)}; dropping colors")
            colors = None
    if normals is not None:
        normals = np.asarray(normals).reshape(-1, 3).astype(np.float32)
        if len(normals) != len(xyz):
            print(f"[warn] normal count {len(normals)} != coord count {len(xyz)}; dropping normals")
            normals = None

    if instance is not None:
        instance = np.asarray(instance).reshape(-1)
        if len(instance) != len(xyz):
            print(f"[warn] instance count {len(instance)} != coord count {len(xyz)}; dropping instances")
            instance = None
        # else:
        #     print_label_stats("Instances", instance)

    if segment20 is not None:
        segment20 = np.asarray(segment20).reshape(-1)
        if len(segment20) != len(xyz):
            print(f"[warn] seg20 count {len(segment20)} != coord count {len(xyz)}; dropping seg20")
            segment20 = None
        # else:
        #     print_label_stats("Segment20", segment20)

    if segment200 is not None:
        segment200 = np.asarray(segment200).reshape(-1)
        if len(segment200) != len(xyz):
            print(f"[warn] segment200 count {len(segment200)} != coord count {len(xyz)}; dropping segment200")
            segment200 = None
        # else:
        #     print_label_stats("Segment200", segment200)

    # Heuristic mm -> m
    if np.linalg.norm(np.nanmax(np.abs(xyz), axis=0)) > 100:
        print("[info] Detected large magnitudes; scaling XYZ by 1/1000 (mm -> m)")
        xyz = xyz / 1000.0

    # Build base point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # Normals if needed (better shading)
    if estimate_normals or normals is None:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.normalize_normals()

    # Optional downsampling
    # label_arrays = {"instance": instance, "segment20": segment20, "segment200": segment200}
    # down_cache = None
    # if downsample_voxel:
    #     pcd, down_cache = downsample_with_label_propagation(
    #         pcd,
    #         orig_xyz=xyz,
    #         orig_colors=colors,
    #         lbl_dict=label_arrays if any(v is not None for v in label_arrays.values()) else None,
    #         voxel_size=downsample_voxel,
    #         fast=fast_down,
    #     )
    #     if down_cache is not None:
    #         if down_cache["rgb"] is not None:
    #             colors = down_cache["rgb"]
    #         if down_cache["instance"] is not None:
    #             instance = down_cache["instance"]
    #         if down_cache["segment20"] is not None:
    #             segment20 = down_cache["segment20"]
    #         if down_cache["segment200"] is not None:
    #             segment200 = down_cache["segment200"]
    #     # refresh references after downsampling
    #     xyz = np.asarray(pcd.points)
    #     if colors is not None:
    #         pcd.colors = o3d.utility.Vector3dVector(colors)

    # ----------------- Instance bounding boxes (precompute) -----------------
    boxes, box_ids = build_instance_bboxes(xyz, instance)
    if boxes:
        print(f"[info] Prepared {len(boxes)} instance bounding boxes.")
    else:
        if instance is not None:
            print("[warn] No instance boxes were created (empty or degenerate).")

    # ----------------- Prepare color modes -----------------
    # We precompute color arrays for fast toggling.
    color_modes = []  # list of (name, color_array)
    base_rgb = np.asarray(pcd.colors).copy() if len(pcd.colors) == len(xyz) else None

    if base_rgb is not None:
        color_modes.append(("RGB", base_rgb))

    if instance is not None:
        inst_cols = hash_colors_for_labels(instance, seed=123)
        color_modes.append(("Instance", inst_cols))

    if segment20 is not None:
        segment20_cols = palette_colors_for_labels(segment20, SCANNET20_COLORS)
        color_modes.append(("Segment20", segment20_cols))

    if segment200 is not None:
        segment200_cols = hash_colors_for_labels(segment200, seed=200)
        color_modes.append(("Segment200", segment200_cols))

    if not color_modes:
        # Ensure we have *some* colors (normals-based or fixed gray)
        fallback = np.full((len(xyz), 3), 0.7, dtype=np.float32)
        pcd.colors = o3d.utility.Vector3dVector(fallback)
        color_modes.append(("Gray", fallback))
    else:
        # Start with the first available mode
        pcd.colors = o3d.utility.Vector3dVector(color_modes[0][1])

    # World axis gizmo
    
    
    print("\n[controls]")
    print("  1 → RGB (if available)")
    print("  2 → Instance (if available)")
    print("  3 → Segment20 (if available)")
    print("  4 → Segment200 (if available)")
    print("  c → Cycle through color modes")
    print("  b → Toggle instance bounding boxes")
    print("  x → Toggle world axis")
    print("  + / = → Increase point size")
    print("  - / _ → Decrease point size")
    print("  h → Print help and label stats\n")

    # ----------------- Key callbacks + state -----------------
    state = {
        "mode_idx": 0,
        "boxes_on": False,
        "axis_on": False,
        "pt_size": float(pt_size_init),
    }

    def set_mode(vis, idx):
        idx = max(0, min(idx, len(color_modes) - 1))
        state["mode_idx"] = idx
        name, cols = color_modes[idx]
        pcd.colors = o3d.utility.Vector3dVector(cols)
        vis.update_geometry(pcd)
        vis.update_renderer()
        print(f"[mode] -> {name}")
        return False

    def cb_1(vis):  # RGB
        for i, (name, _) in enumerate(color_modes):
            if name.lower() == "rgb":
                return set_mode(vis, i)
        print("[info] RGB colors not available.")
        return False

    def cb_2(vis):  # Instance
        for i, (name, _) in enumerate(color_modes):
            if name.lower() == "instance":
                return set_mode(vis, i)
        print("[info] Instance labels not available.")
        return False

    def cb_3(vis):  # Seg20
        for i, (name, _) in enumerate(color_modes):
            if name.lower() == "segment20":
                return set_mode(vis, i)
        print("[info] Segment20 labels not available.")
        return False

    def cb_4(vis):  # Seg200
        for i, (name, _) in enumerate(color_modes):
            if name.lower() == "segment200":
                return set_mode(vis, i)
        print("[info] Segment200 labels not available.")
        return False

    def cb_c(vis):  # cycle
        return set_mode(vis, (state["mode_idx"] + 1) % len(color_modes))

    def cb_h(vis):
        print("\n[controls]")
        print("  1 → RGB (if available)")
        print("  2 → Instance (if available)")
        print("  3 → Segment20 (if available)")
        print("  4 → Segment200 (if available)")
        print("  c → Cycle through color modes")
        print("  b → Toggle instance bounding boxes")
        print("  x → Toggle world axis")
        print("  + / = → Increase point size")
        print("  - / _ → Decrease point size")
        print("  h → Print this help\n")
        # Uncomment if you want stats:
        # if instance is not None:  print_label_stats("Instances", instance)
        # if segment20 is not None: print_label_stats("Segment20", segment20)
        # if segment200 is not None: print_label_stats("Segment200", segment200)
        return False
        

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

    def cb_x(vis):
      axis_geoms = make_thin_axis(size=axis_size, thickness=axis_thickness)
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

    # Key map
    key_to_callback = {
        ord("1"): cb_1,
        ord("2"): cb_2,
        ord("3"): cb_3,
        ord("4"): cb_4,
        ord("c"): cb_c,
        ord("C"): cb_c,
        ord("h"): cb_h,
        ord("H"): cb_h,
        ord("b"): cb_b,
        ord("B"): cb_b,
        ord("+"): cb_plus,  
        ord("="): cb_plus,
        ord("-"): cb_minus,
        ord("_"): cb_minus,
        ord("x"): cb_x,
        ord("X"): cb_x,
    }

    # ----------------- Visualize -----------------
    # Use a visualizer so we can set initial point size and (optionally) axis.
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"ScanNet Viewer - {scene_dir}")
    vis.add_geometry(pcd)

    # Register callbacks
    for k, cb in key_to_callback.items():
        vis.register_key_callback(k, cb)

    # Initial render options
    ro = vis.get_render_option()
    ro.point_size = float(pt_size_init)


    vis.run()
    vis.destroy_window()

    # ----------------- Save (if requested) -----------------
    if save_ply:
        os.makedirs(os.path.dirname(save_ply) or ".", exist_ok=True)
        # Save with the currently visible colors
        ok = o3d.io.write_point_cloud(save_ply, pcd)
        if ok:
            print(f"[info] Saved: {save_ply}")
        else:
            print(f"[error] Failed to save: {save_ply}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", help="Folder with coord.npy, color.npy, (optional) normal.npy/instance.npy/segment20.npy/segment200.npy")
    #ap.add_argument("--down", type=float, default=None, help="Voxel size for downsampling in meters, e.g. 0.01")
    ap.add_argument("--save", type=str, default=None, help="Output PLY path to save the *currently visible* point cloud")
    ap.add_argument("--est", action="store_true", help="Force normal estimation")
    ap.add_argument("--fast-down", action="store_true",
                    help="Faster downsampling (does not reassign labels; instances/segments may look wrong)")
    ap.add_argument("--pt", type=float, default=2.0,
                    help="Initial point size in pixels (default: 2.0)")
    ap.add_argument("--axis", action="store_true",
                    help="Show world axis gizmo at start (toggle with 'x')")
    ap.add_argument("--axis-size", type=float, default=10,
                    help="Axis gizmo size in scene units (default: 10)")
    ap.add_argument("--axis-thickness", type=float, default=0.002,
                help="Axis radius as a fraction of axis size (default: 0.002)")
    args = ap.parse_args()
    main(args.scene_dir, args.save, args.est, args.fast_down,
         args.pt, args.axis, args.axis_size,args.axis_thickness)
