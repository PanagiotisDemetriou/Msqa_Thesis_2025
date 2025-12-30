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