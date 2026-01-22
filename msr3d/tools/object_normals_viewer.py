import os
import argparse
import torch
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf
from data.datasets.scannet_base import ScanNetBase


def load_normals_cache(normals_dir: str, scan_id: str):
    path = os.path.join(normals_dir, f"{scan_id}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Normals cache not found: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if "obj_normals_list" not in cache:
        raise KeyError(f"Normals cache missing 'obj_normals_list': {path}")
    return cache


def visualize_object_with_normals(
    xyz: np.ndarray,
    rgb01: np.ndarray | None,
    normals: np.ndarray,
    voxel: float = 0.0,
    point_size: float = 3.0,
    show_normals: bool = True,
):
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)

    if xyz.shape[0] != normals.shape[0]:
        raise ValueError(f"Points/normals mismatch: points={xyz.shape[0]} normals={normals.shape[0]}")

    # Normalize normals (should already be unit, but safe)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / nlen

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    if rgb01 is not None:
        rgb01 = np.asarray(rgb01, dtype=np.float32).reshape(-1, 3)
        if rgb01.shape[0] == xyz.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb01, 0.0, 1.0))
        else:
            # fallback color
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # Optional voxel downsample for speed
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel))

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Object Normals Viewer")
    vis.add_geometry(pcd)

    ro = vis.get_render_option()
    ro.point_size = float(point_size)
    ro.point_show_normal = bool(show_normals)

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Visualize cached normals for a single object from a ScanNet scene.")
    parser.add_argument("--cfg", default="msr3d/configs/data.yaml", help="Path to msr3d data.yaml")
    parser.add_argument("--split", default="train", help="Dataset split (train/val/test)")
    parser.add_argument("--scan_id", default="scene0000_00", help="ScanNet scene id, e.g. scene0000_00")
    parser.add_argument("--obj_idx", type=int, default=0, help="Object index within the scene")
    parser.add_argument("--normals_dir",
                        default="/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals/",
                        help="Directory containing normals cache files <scan_id>.pth")
    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (0 disables)")
    parser.add_argument("--point_size", type=float, default=3.0, help="Open3D point size")
    parser.add_argument("--no_normals", action="store_true", help="Do not show normals (points only)")
    args = parser.parse_args()

    # Loader-based points (guaranteed to align with your normals cache)
    cfg = OmegaConf.load(args.cfg)
    loader = ScanNetBase(cfg, split=args.split)

    _, one_scan = loader._load_one_scan(args.scan_id, load_inst_info=True, load_pc_info=True)
    obj_pcds_list = one_scan.get("obj_pcds", None)
    if obj_pcds_list is None:
        raise RuntimeError(f"{args.scan_id}: loader returned no 'obj_pcds'")

    if args.obj_idx < 0 or args.obj_idx >= len(obj_pcds_list):
        raise IndexError(f"obj_idx out of range: {args.obj_idx} (scene has {len(obj_pcds_list)} objects)")

    obj = obj_pcds_list[args.obj_idx]  # (Ni,6) xyz rgb
    xyz = obj[:, :3]
    rgb01 = obj[:, 3:6].astype(np.float32) / 255.0

    # Cached normals
    cache = load_normals_cache(args.normals_dir, args.scan_id)
    obj_normals_list = cache["obj_normals_list"]

    if args.obj_idx >= len(obj_normals_list):
        raise IndexError(
            f"Normals cache has only {len(obj_normals_list)} objects, requested obj_idx={args.obj_idx}"
        )

    normals = obj_normals_list[args.obj_idx]
    print(f"[info] scan_id={args.scan_id} obj_idx={args.obj_idx} points={xyz.shape[0]}")
    print(f"[info] normals meta={cache.get('meta', None)}")

    visualize_object_with_normals(
        xyz=xyz,
        rgb01=rgb01,
        normals=normals,
        voxel=args.voxel,
        point_size=args.point_size,
        show_normals=(not args.no_normals),
    )


if __name__ == "__main__":
    main()

