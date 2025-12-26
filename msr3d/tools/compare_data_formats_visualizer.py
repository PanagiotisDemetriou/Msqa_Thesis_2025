# import argparse
# import os
# from collections import Counter

# import numpy as np
# import torch
# import open3d as o3d


# def normalize_old_colors(colors):
#     """
#     Old loader sometimes provides uint8 [0..255] or float in [-1..1] or [0..1].
#     Return float32 in [0..1].
#     """
#     c = np.asarray(colors)
#     if c.dtype != np.float32:
#         c = c.astype(np.float32)

#     cmin, cmax = float(c.min()), float(c.max())

#     # Likely uint8-ish
#     if cmax > 1.5:
#         c = c / 255.0
#         return np.clip(c, 0.0, 1.0)

#     # Likely [-1, 1]
#     if cmin < 0.0:
#         c = (c + 1.0) / 2.0
#         return np.clip(c, 0.0, 1.0)

#     # Already [0,1]
#     return np.clip(c, 0.0, 1.0)


# def hash_colors_for_labels(labels, seed=123):
#     """Deterministic random colors for arbitrary label IDs."""
#     rng = np.random.default_rng(seed)
#     uniq = np.unique(labels)
#     table = {}
#     for u in uniq:
#         col = rng.random(3) * 0.8 + 0.2
#         table[int(u)] = col.astype(np.float32)
#     return np.array([table[int(x)] for x in labels], dtype=np.float32)


# def build_instance_bboxes(xyz, instance_labels, seed=123):
#     xyz = np.asarray(xyz, dtype=np.float32)
#     inst = np.asarray(instance_labels).reshape(-1)

#     uniq = np.unique(inst)
#     cols = hash_colors_for_labels(uniq, seed=seed)
#     col_tab = {int(u): c for u, c in zip(uniq, cols)}

#     bboxes = []
#     for u in uniq:
#         mask = inst == u
#         if mask.sum() < 2:
#             continue
#         pts = xyz[mask]
#         bb = o3d.geometry.AxisAlignedBoundingBox(
#             min_bound=pts.min(0), max_bound=pts.max(0)
#         )
#         bb.color = col_tab[int(u)].tolist()
#         bboxes.append(bb)
#     return bboxes


# def print_label_stats(name, labels):
#     labels = np.asarray(labels).reshape(-1)
#     c = Counter(labels.tolist())
#     top = c.most_common(10)
#     more = max(0, len(c) - 10)
#     print(f"[stats] {name}: {len(c)} unique labels; top 10:")
#     for k, v in top:
#         print(f"   ID {k:>4}: {v}")
#     if more:
#         print(f"   ... and {more} more")


# def load_old_scene(old_pth_path):
#     pcd_data = torch.load(old_pth_path, weights_only=False)
#     points = np.asarray(pcd_data[0], dtype=np.float32)
#     colors = normalize_old_colors(pcd_data[1])
#     instance_labels = np.asarray(pcd_data[-1]).reshape(-1)

#     assert points.shape[0] == colors.shape[0] == instance_labels.shape[0]
#     return points, colors, instance_labels


# def load_new_scene(new_path):
#     """
#     Supports either:
#       A) obj_pcds tensor: (1, N, P, 6) or (N, P, 6)
#       B) point_data dict: with keys 'coord' (Npts,3) and 'feat' (Npts,C>=6)
#     Returns:
#       points (Npts,3), colors (Npts,3) in [0,1], instance_labels (Npts,)
#     """
#     obj = torch.load(new_path, weights_only=False)

#     # Case B: point_data dict
#     if isinstance(obj, dict) and ("coord" in obj) and ("feat" in obj):
#         coord = obj["coord"]
#         feat = obj["feat"]
#         if torch.is_tensor(coord):
#             coord = coord.detach().cpu()
#         if torch.is_tensor(feat):
#             feat = feat.detach().cpu()

#         points = coord.numpy().astype(np.float32)
#         feat_np = feat.numpy().astype(np.float32)
#         if feat_np.shape[1] < 6:
#             raise ValueError("point_data['feat'] must have at least 6 channels (xyzrgb-like).")
#         rgb = feat_np[:, 3:6]

#         # If rgb is [-1,1], map to [0,1]
#         rgb = normalize_old_colors(rgb)

#         # If no instance info, label everything 0
#         inst = np.zeros((points.shape[0],), dtype=np.int32)
#         return points, rgb, inst

#     # Case A: obj_pcds tensor
#     if torch.is_tensor(obj):
#         t = obj.detach().cpu()
#         if t.ndim == 4:
#             # (B, N, P, C)
#             if t.shape[0] != 1:
#                 raise ValueError("Expected B=1 for obj_pcds, got shape: " + str(tuple(t.shape)))
#             t = t[0]  # (N,P,C)
#         elif t.ndim != 3:
#             raise ValueError("Unsupported tensor shape for obj_pcds: " + str(tuple(t.shape)))

#         N, P, C = t.shape
#         if C < 6:
#             raise ValueError("obj_pcds must have at least 6 channels (xyz+rgb).")

#         pts = t[:, :, :3].reshape(-1, 3).numpy().astype(np.float32)
#         rgb = t[:, :, 3:6].reshape(-1, 3).numpy().astype(np.float32)
#         rgb = normalize_old_colors(rgb)

#         inst = np.repeat(np.arange(N, dtype=np.int32), P)
#         return pts, rgb, inst

#     raise ValueError("Unsupported new format in file: expected dict(coord/feat) or tensor(obj_pcds).")


# def make_pcd(points, colors):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32))
#     pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float32))
#     return pcd


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--old", required=True, help="Path to old scene .pth (points, colors, instance_labels).")
#     ap.add_argument("--new", required=True, help="Path to new-format .pth (obj_pcds tensor or point_data dict).")
#     ap.add_argument("--point_size", type=float, default=2.0)
#     ap.add_argument("--gap", type=float, default=1.5, help="Multiplier for spacing between old and new.")
#     args = ap.parse_args()

#     old_points, old_colors, old_inst = load_old_scene(args.old)
#     new_points, new_colors, new_inst = load_new_scene(args.new)

#     print("[info] Old:", old_points.shape, old_colors.shape, old_inst.shape)
#     print("[info] New:", new_points.shape, new_colors.shape, new_inst.shape)
#     print_label_stats("Old instances", old_inst)
#     print_label_stats("New instances", new_inst)

#     old_pcd = make_pcd(old_points, old_colors)
#     new_pcd = make_pcd(new_points, new_colors)

#     # Compute translation so "new" appears to the right of "old"
#     old_bb = old_pcd.get_axis_aligned_bounding_box()
#     extent = old_bb.get_extent()
#     shift = float(np.linalg.norm(extent)) * args.gap
#     new_pcd.translate((shift, 0.0, 0.0))

#     # Precompute instance-colored versions
#     old_inst_cols = hash_colors_for_labels(old_inst, seed=123)
#     new_inst_cols = hash_colors_for_labels(new_inst, seed=123)

#     # Bounding boxes (optional toggle)
#     old_boxes = build_instance_bboxes(old_points, old_inst, seed=123)
#     new_boxes = build_instance_bboxes(new_points + np.array([shift, 0.0, 0.0], dtype=np.float32), new_inst, seed=123)

#     # Visualizer state
#     state = {
#         "old_mode": "rgb",  # rgb | inst
#         "new_mode": "rgb",
#         "old_boxes": False,
#         "new_boxes": False,
#         "point_size": float(args.point_size),
#     }

#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window(window_name="Old (left) vs New (right) comparison", width=1600, height=900)

#     vis.add_geometry(old_pcd)
#     vis.add_geometry(new_pcd)

#     ro = vis.get_render_option()
#     ro.point_size = state["point_size"]

#     print("\n[controls]")
#     print("  1: Old RGB")
#     print("  2: Old Instance colors")
#     print("  3: New RGB")
#     print("  4: New Instance colors")
#     print("  b: Toggle Old bounding boxes")
#     print("  n: Toggle New bounding boxes")
#     print("  + / =: Increase point size")
#     print("  - / _: Decrease point size")
#     print("  q / ESC: Quit\n")

#     def update_old_colors():
#         if state["old_mode"] == "rgb":
#             old_pcd.colors = o3d.utility.Vector3dVector(old_colors)
#         else:
#             old_pcd.colors = o3d.utility.Vector3dVector(old_inst_cols)
#         vis.update_geometry(old_pcd)
#         vis.update_renderer()

#     def update_new_colors():
#         if state["new_mode"] == "rgb":
#             new_pcd.colors = o3d.utility.Vector3dVector(new_colors)
#         else:
#             new_pcd.colors = o3d.utility.Vector3dVector(new_inst_cols)
#         vis.update_geometry(new_pcd)
#         vis.update_renderer()

#     def toggle_old_boxes():
#         if not state["old_boxes"]:
#             for bb in old_boxes:
#                 vis.add_geometry(bb)
#             state["old_boxes"] = True
#             print("[mode] Old boxes: ON")
#         else:
#             for bb in old_boxes:
#                 vis.remove_geometry(bb, reset_bounding_box=False)
#             state["old_boxes"] = False
#             print("[mode] Old boxes: OFF")
#         vis.update_renderer()

#     def toggle_new_boxes():
#         if not state["new_boxes"]:
#             for bb in new_boxes:
#                 vis.add_geometry(bb)
#             state["new_boxes"] = True
#             print("[mode] New boxes: ON")
#         else:
#             for bb in new_boxes:
#                 vis.remove_geometry(bb, reset_bounding_box=False)
#             state["new_boxes"] = False
#             print("[mode] New boxes: OFF")
#         vis.update_renderer()

#     def apply_point_size():
#         ro = vis.get_render_option()
#         ro.point_size = float(max(1.0, min(state["point_size"], 50.0)))
#         vis.update_renderer()

#     # Key callbacks
#     def cb_1(_): state.update(old_mode="rgb"); update_old_colors(); print("[mode] Old -> RGB"); return False
#     def cb_2(_): state.update(old_mode="inst"); update_old_colors(); print("[mode] Old -> Instance"); return False
#     def cb_3(_): state.update(new_mode="rgb"); update_new_colors(); print("[mode] New -> RGB"); return False
#     def cb_4(_): state.update(new_mode="inst"); update_new_colors(); print("[mode] New -> Instance"); return False
#     def cb_b(_): toggle_old_boxes(); return False
#     def cb_n(_): toggle_new_boxes(); return False
#     def cb_plus(_): state["point_size"] *= 1.25; apply_point_size(); print(f"[mode] point size={state['point_size']:.2f}"); return False
#     def cb_minus(_): state["point_size"] /= 1.25; apply_point_size(); print(f"[mode] point size={state['point_size']:.2f}"); return False

#     vis.register_key_callback(ord("1"), cb_1)
#     vis.register_key_callback(ord("2"), cb_2)
#     vis.register_key_callback(ord("3"), cb_3)
#     vis.register_key_callback(ord("4"), cb_4)
#     vis.register_key_callback(ord("b"), cb_b)
#     vis.register_key_callback(ord("B"), cb_b)
#     vis.register_key_callback(ord("n"), cb_n)
#     vis.register_key_callback(ord("N"), cb_n)
#     vis.register_key_callback(ord("+"), cb_plus)
#     vis.register_key_callback(ord("="), cb_plus)
#     vis.register_key_callback(ord("-"), cb_minus)
#     vis.register_key_callback(ord("_"), cb_minus)

#     vis.run()
#     vis.destroy_window()


# if __name__ == "__main__":
#     main()
import argparse
import numpy as np
import torch
import open3d as o3d


def normalize_colors_any(colors):
    """
    Accepts uint8 [0..255], float [0..1], or float [-1..1].
    Returns float32 in [0..1].
    """
    c = np.asarray(colors)
    if c.dtype != np.float32:
        c = c.astype(np.float32)

    cmin, cmax = float(c.min()), float(c.max())

    if cmax > 1.5:          # likely uint8-like
        c = c / 255.0
    elif cmin < 0.0:        # likely [-1,1]
        c = (c + 1.0) / 2.0

    return np.clip(c, 0.0, 1.0).astype(np.float32)


def load_old_scene(old_pth):
    pcd_data = torch.load(old_pth, weights_only=False)
    points = np.asarray(pcd_data[0], dtype=np.float32)
    colors = normalize_colors_any(pcd_data[1])
    instance_labels = np.asarray(pcd_data[-1]).reshape(-1).astype(np.int64)

    if not (len(points) == len(colors) == len(instance_labels)):
        raise ValueError("Old scene arrays have inconsistent lengths.")
    return points, colors, instance_labels


def load_new_obj_pcds(new_pth):
    obj = torch.load(new_pth, weights_only=False)

    if not torch.is_tensor(obj):
        raise ValueError("Expected NEW file to be a tensor obj_pcds (B,N,P,C) or (N,P,C).")

    t = obj.detach().cpu()
    if t.ndim == 4:
        # (B,N,P,C) -> assume B=1
        if t.shape[0] != 1:
            raise ValueError(f"Expected B=1 in new obj_pcds, got {tuple(t.shape)}")
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Unsupported obj_pcds shape: {tuple(t.shape)}")

    if t.shape[2] < 6:
        raise ValueError("obj_pcds must have at least 6 channels (xyz+rgb).")

    return t  # (N,P,C)


def make_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float32))
    return pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="Old scene .pth path (points, colors, instance_labels).")
    ap.add_argument("--new", required=True, help="New obj_pcds .pth path (tensor).")
    ap.add_argument("--obj_idx", type=int, required=True, help="Instance id / object index to compare.")
    ap.add_argument("--point_size", type=float, default=3.0)
    ap.add_argument("--gap", type=float, default=2.0, help="Spacing multiplier between old/new objects.")
    ap.add_argument("--min_old_points", type=int, default=50, help="Warn if old object has too few points.")
    args = ap.parse_args()

    old_points, old_colors, old_inst = load_old_scene(args.old)
    obj_pcds = load_new_obj_pcds(args.new)  # (N,P,C)

    N, P, C = obj_pcds.shape
    if not (0 <= args.obj_idx < N):
        raise ValueError(f"obj_idx {args.obj_idx} out of range for new obj_pcds with N={N}")

    # ----- Extract OLD object by instance label -----
    mask = (old_inst == args.obj_idx)
    if mask.sum() == 0:
        raise ValueError(
            f"No points in OLD scene with instance label == {args.obj_idx}. "
            "This means your obj_idx does not correspond to that instance id in the old scene."
        )

    old_obj_pts = old_points[mask]
    old_obj_cols = old_colors[mask]

    if old_obj_pts.shape[0] < args.min_old_points:
        print(f"[warn] Old object {args.obj_idx} has only {old_obj_pts.shape[0]} points.")

    # ----- Extract NEW object by index -----
    new_obj = obj_pcds[args.obj_idx]  # (P,C)
    new_obj_pts = new_obj[:, :3].numpy().astype(np.float32)
    new_obj_cols = normalize_colors_any(new_obj[:, 3:6].numpy().astype(np.float32))

    # ----- Build point clouds -----
    old_pcd = make_pcd(old_obj_pts, old_obj_cols)
    new_pcd = make_pcd(new_obj_pts, new_obj_cols)

    # ----- Place side-by-side -----
    old_bb = old_pcd.get_axis_aligned_bounding_box()
    extent = old_bb.get_extent()
    shift = float(np.linalg.norm(extent)) * float(args.gap)
    if not np.isfinite(shift) or shift < 1e-6:
        shift = 1.0
    new_pcd.translate((shift, 0.0, 0.0))

    # ----- Visualize -----
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Compare object {args.obj_idx}: OLD (left) vs NEW (right)", width=1400, height=900)

    vis.add_geometry(old_pcd)
    vis.add_geometry(new_pcd)

    ro = vis.get_render_option()
    ro.point_size = float(args.point_size)

    print("\n[info] Showing object", args.obj_idx)
    print("[info] OLD points:", old_obj_pts.shape[0], " NEW points:", new_obj_pts.shape[0])
    print("[info] If NEW is normalized per-object, shapes should match but global placement will differ.")
    print("[info] Close window with Q or ESC.\n")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()


# ./msr3d/tools/compare_visualizer.sh   -p /home/panagiotis/miniconda3/envs/pointcept-torch2.5.0-cu12.4/bin/python   --old /mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0000_00.pth   --new /home/panagiotis/msqa/Msqa_Thesis_2025/msr3d/tools/scene0000_00_new_obj_pcds.pth   --obj_idx 2