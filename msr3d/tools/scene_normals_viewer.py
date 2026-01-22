# #!/usr/bin/env python3
# """
# ScanNet Scene Normals Viewer (from MSR3D normals cache) + Normal-direction coloring

# Goal:
# - Load the dense scene point cloud from ScanNetBase (concatenated obj_pcds)
# - Load per-object normals from your normals cache: <normals_dir>/<scan_id>.pth with "obj_normals_list"
# - Concatenate per-object normals to align with the concatenated scene points
# - Visualize the full scene with colors indexed by normal direction (like typical Open3D normal visualization):
#     RGB = (n + 1) / 2

# This is designed to let you compare:
# 1) SHS-Net normals visualization (your earlier robust script)
# vs
# 2) Cached normals visualization (this script) on the same *dense* scene points.

# Typical use:
#   python msr3d/tools/scene_normals_viewer_cache.py \
#     --scan_id scene0000_00 \
#     --cfg msr3d/configs/data.yaml --split train \
#     --normals_dir /mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals/ \
#     --color_by_normals \
#     --voxel 0.02

# Notes:
# - If your cache normals have sign ambiguity (n vs -n), colors will flip accordingly.
#   You can optionally enforce a consistent orientation with --orient_viewpoint.
# """

# import os
# import argparse
# import torch
# import numpy as np
# import open3d as o3d
# from omegaconf import OmegaConf

# from data.datasets.scannet_base import ScanNetBase


# def load_normals_cache(normals_dir: str, scan_id: str):
#     path = os.path.join(normals_dir, f"{scan_id}.pth")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Normals cache not found: {path}")
#     cache = torch.load(path, map_location="cpu", weights_only=False)
#     if "obj_normals_list" not in cache:
#         raise KeyError(f"Normals cache missing 'obj_normals_list': {path}")
#     return cache


# def normalize_normals(normals: np.ndarray) -> np.ndarray:
#     normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
#     nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
#     return normals / nlen


# def normals_to_rgb01(normals: np.ndarray) -> np.ndarray:
#     """
#     Direction-to-color mapping:
#       RGB = (n + 1) / 2  with n in [-1,1]
#     """
#     n = normalize_normals(normals)
#     rgb = 0.5 * (n + 1.0)
#     return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


# def orient_normals_toward_viewpoint(xyz: np.ndarray, normals: np.ndarray, viewpoint: np.ndarray) -> np.ndarray:
#     """
#     Optional: flip normals so they generally face a viewpoint to reduce sign-flip artifacts.
#     """
#     xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
#     n = normalize_normals(normals)
#     vp = np.asarray(viewpoint, dtype=np.float32).reshape(1, 3)
#     v = vp - xyz
#     flip = (np.sum(n * v, axis=1) < 0.0).astype(np.float32).reshape(-1, 1)
#     return n * (1.0 - 2.0 * flip)


# def load_scene_xyz_rgb(one_scan: dict) -> tuple[np.ndarray, np.ndarray | None]:
#     """
#     Build dense scene by concatenating arrays in one_scan["obj_pcds"].
#     Each entry is expected to be (Ni, >=3) and often (Ni, 6) = xyzrgb.
#     """
#     obj_pcds = one_scan.get("obj_pcds", None)
#     if obj_pcds is None or len(obj_pcds) == 0:
#         raise RuntimeError("one_scan has no 'obj_pcds' to build scene point cloud.")

#     parts = []
#     for i, obj in enumerate(obj_pcds):
#         arr = np.asarray(obj)
#         if arr.ndim != 2 or arr.shape[1] < 3:
#             raise RuntimeError(f"obj_pcds[{i}] has invalid shape: {arr.shape}")
#         parts.append(arr)

#     scene = np.concatenate(parts, axis=0)
#     xyz = scene[:, :3].astype(np.float32)

#     rgb01 = None
#     if scene.shape[1] >= 6:
#         rgb = scene[:, 3:6].astype(np.float32)
#         if rgb.size > 0 and rgb.max() > 1.5:
#             rgb01 = np.clip(rgb / 255.0, 0.0, 1.0)
#         else:
#             rgb01 = np.clip(rgb, 0.0, 1.0)

#     return xyz, rgb01


# def load_scene_normals_from_cache(one_scan: dict, cache: dict, strict: bool = False):
#     """
#     Robustly build a scene point set + normals from cache.

#     If obj_pcds and obj_normals_list lengths match: concatenate all.
#     If they don't: concatenate only the overlapping prefix (K=min(...)) unless strict=True.

#     Returns:
#       xyz_cat: (M,3)
#       rgb01_cat: (M,3) or None
#       normals_cat: (M,3)
#     """
#     obj_pcds = one_scan.get("obj_pcds", None)
#     if obj_pcds is None or len(obj_pcds) == 0:
#         raise RuntimeError("one_scan has no 'obj_pcds'")

#     obj_normals_list = cache.get("obj_normals_list", None)
#     if obj_normals_list is None or len(obj_normals_list) == 0:
#         raise KeyError("cache missing or empty 'obj_normals_list'")

#     n_pcds = len(obj_pcds)
#     n_norm = len(obj_normals_list)

#     if n_pcds != n_norm:
#         msg = f"obj_pcds entries={n_pcds} != cache normals entries={n_norm}"
#         if strict:
#             raise ValueError(msg)
#         print(f"[warn] {msg}. Using only overlapping prefix for visualization.")

#     K = min(n_pcds, n_norm)

#     xyz_parts = []
#     rgb_parts = []
#     have_rgb = True
#     normals_parts = []

#     for i in range(K):
#         pts = np.asarray(obj_pcds[i])
#         nrm = np.asarray(obj_normals_list[i])

#         if pts.ndim != 2 or pts.shape[1] < 3:
#             raise RuntimeError(f"obj_pcds[{i}] invalid shape: {pts.shape}")
#         if nrm.ndim != 2 or nrm.shape[1] != 3:
#             raise RuntimeError(f"obj_normals_list[{i}] invalid shape: {nrm.shape}")
#         if pts.shape[0] != nrm.shape[0]:
#             raise ValueError(f"Mismatch at entry {i}: points={pts.shape[0]} normals={nrm.shape[0]}")

#         xyz_parts.append(pts[:, :3].astype(np.float32, copy=False))
#         normals_parts.append(nrm.astype(np.float32, copy=False))

#         if pts.shape[1] >= 6:
#             rgb = pts[:, 3:6].astype(np.float32, copy=False)
#             if rgb.max() > 1.5:
#                 rgb = np.clip(rgb / 255.0, 0.0, 1.0)
#             else:
#                 rgb = np.clip(rgb, 0.0, 1.0)
#             rgb_parts.append(rgb)
#         else:
#             have_rgb = False

#     xyz_cat = np.concatenate(xyz_parts, axis=0)
#     normals_cat = np.concatenate(normals_parts, axis=0)

#     rgb01_cat = None
#     if have_rgb and len(rgb_parts) == K:
#         rgb01_cat = np.concatenate(rgb_parts, axis=0)

#     return xyz_cat, rgb01_cat, normals_cat



# def visualize_scene(
#     xyz: np.ndarray,
#     rgb01: np.ndarray | None,
#     normals: np.ndarray | None,
#     *,
#     voxel: float = 0.0,
#     point_size: float = 2.0,
#     show_normals: bool = True,
#     color_by_normals: bool = True,
#     keep_scene_rgb: bool = False,
#     orient_viewpoint: np.ndarray | None = None,
#     window_name: str = "Scene Normals Viewer (Cache)",
# ):
#     xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)

#     n_vis = None
#     if normals is not None:
#         n_vis = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
#         if n_vis.shape[0] != xyz.shape[0]:
#             raise ValueError(f"Points/normals mismatch: points={xyz.shape[0]} normals={n_vis.shape[0]}")
#         n_vis = normalize_normals(n_vis)

#         if orient_viewpoint is not None:
#             n_vis = orient_normals_toward_viewpoint(xyz, n_vis, orient_viewpoint)

#         pcd.normals = o3d.utility.Vector3dVector(n_vis)

#     # Colors: normals direction index (default) unless overridden
#     if color_by_normals and (n_vis is not None) and (not keep_scene_rgb):
#         pcd.colors = o3d.utility.Vector3dVector(normals_to_rgb01(n_vis))
#     elif rgb01 is not None and np.asarray(rgb01).shape[0] == xyz.shape[0]:
#         pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb01, 0.0, 1.0))
#     else:
#         pcd.paint_uniform_color([0.7, 0.7, 0.7])

#     if voxel and voxel > 0:
#         pcd = pcd.voxel_down_sample(voxel_size=float(voxel))

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=window_name)
#     vis.add_geometry(pcd)

#     ro = vis.get_render_option()
#     ro.point_size = float(point_size)
#     ro.point_show_normal = bool(show_normals and (n_vis is not None))

#     vis.run()
#     vis.destroy_window()


# def main():
#     parser = argparse.ArgumentParser(
#         description="Visualize dense ScanNet scene normals from MSR3D normals cache with direction-based coloring."
#     )
#     parser.add_argument("--cfg", default="msr3d/configs/data.yaml", help="Path to msr3d data.yaml")
#     parser.add_argument("--split", default="train", help="Dataset split (train/val/test)")
#     parser.add_argument("--scan_id", default="scene0000_00", help="ScanNet scene id, e.g. scene0000_00")

#     parser.add_argument(
#         "--normals_dir",
#         default="/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals/",
#         help="Directory containing normals cache files <scan_id>.pth",
#     )

#     parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (0 disables)")
#     parser.add_argument("--point_size", type=float, default=2.0, help="Open3D point size")
#     parser.add_argument("--no_normals", action="store_true", help="Do not show normal glyphs (points only)")
#     parser.add_argument(
#             "--normal_colors",
#             action="store_true",
#             help="Color by normal direction but do not draw normal glyphs.",
#         )
#     # Coloring controls
#     parser.add_argument(
#         "--color_by_normals",
#         action="store_true",
#         help="Color points by normal direction using RGB=(n+1)/2 (recommended for comparison).",
#     )
#     parser.add_argument(
#         "--keep_scene_rgb",
#         action="store_true",
#         help="If set, do NOT override scene RGB even when --color_by_normals is enabled.",
#     )

#     # Optional normal sign disambiguation
#     parser.add_argument(
#         "--orient_viewpoint",
#         type=float,
#         nargs=3,
#         default=None,
#         metavar=("X", "Y", "Z"),
#         help="Optional viewpoint (x y z) to flip normals toward; reduces sign-flip color inversions.",
#     )

#     args = parser.parse_args()

#     cfg = OmegaConf.load(args.cfg)
#     loader = ScanNetBase(cfg, split=args.split)

#     _, one_scan = loader._load_one_scan(args.scan_id, load_inst_info=True, load_pc_info=True)
#     if one_scan.get("obj_pcds", None) is None:
#         raise RuntimeError(f"{args.scan_id}: loader returned no 'obj_pcds'")

    


#     cache = load_normals_cache(args.normals_dir, args.scan_id)
#     xyz_scene, rgb01_scene, normals_scene = load_scene_normals_from_cache(one_scan, cache, strict=False)
#     #normals_scene = load_scene_normals_from_cache(one_scan, cache)

#     print(f"[info] scan_id={args.scan_id}")
#     print(f"[info] scene points={xyz_scene.shape[0]} scene normals={normals_scene.shape[0]}")
#     print(f"[info] normals meta={cache.get('meta', None)}")
#     # Need normals data if we want normal-direction colors OR normal glyphs
#     use_normals_for_color = args.color_by_normals or getattr(args, "normal_colors", False)

#     # Draw glyphs only if user didn't disable them and isn't in "colors-only" mode
#     draw_normal_glyphs = (not args.no_normals) and (not getattr(args, "normal_colors", False))

#     visualize_scene(
#         xyz=xyz_scene,
#         rgb01=rgb01_scene,
#         normals=normals_scene if (use_normals_for_color or draw_normal_glyphs) else None,
#         voxel=args.voxel,
#         point_size=args.point_size,
#         show_normals=draw_normal_glyphs,
#         color_by_normals=use_normals_for_color,
#         keep_scene_rgb=args.keep_scene_rgb,
#         orient_viewpoint=None if args.orient_viewpoint is None else np.array(args.orient_viewpoint, dtype=np.float32),
#         window_name="PCA/Open3D Normals",
#     )
#     # visualize_scene(
#     #     xyz=xyz_scene,
#     #     rgb01=rgb01_scene,
#     #     normals=None if args.no_normals else normals_scene,
#     #     voxel=args.voxel,
#     #     point_size=args.point_size,
#     #     show_normals=(not args.no_normals),
#     #     color_by_normals=args.color_by_normals,
#     #     keep_scene_rgb=args.keep_scene_rgb,
#     #     orient_viewpoint=None if args.orient_viewpoint is None else np.array(args.orient_viewpoint, dtype=np.float32),
#     #     window_name="PCA/Open3D Normals",
#     # )


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
ScanNet Scene Normals Viewer (from _load_one_scan obj_pcds) + normal-direction coloring

This version does NOT load normals from the cache .pth.
It uses the normals that _load_one_scan() concatenates into obj_pcds (expects Nx9: xyzrgb + normals).

Usage:
  python msr3d/tools/scene_normals_viewer_from_loader.py \
    --scan_id scene0000_00 \
    --cfg msr3d/configs/data.yaml --split train \
    --color_by_normals \
    --voxel 0.02
"""

import argparse
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf

from data.datasets.scannet_base import ScanNetBase


def normalize_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals / nlen


def normals_to_rgb01(normals: np.ndarray) -> np.ndarray:
    n = normalize_normals(normals)
    rgb = 0.5 * (n + 1.0)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def orient_normals_toward_viewpoint(xyz: np.ndarray, normals: np.ndarray, viewpoint: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = normalize_normals(normals)
    vp = np.asarray(viewpoint, dtype=np.float32).reshape(1, 3)
    v = vp - xyz
    flip = (np.sum(n * v, axis=1) < 0.0).astype(np.float32).reshape(-1, 1)
    return n * (1.0 - 2.0 * flip)


def load_scene_from_loader(one_scan: dict, require_normals: bool = True):
    """
    Concatenate one_scan["obj_pcds"] into a dense scene.

    Expects each obj_pcds[i] to be:
      - (Ni, 6) xyzrgb OR
      - (Ni, 9) xyzrgb + normals

    If require_normals=True, enforces that normals exist.
    Returns:
      xyz (N,3), rgb01 (N,3) or None, normals (N,3) or None
    """
    obj_pcds = one_scan.get("obj_pcds", None)
    if obj_pcds is None or len(obj_pcds) == 0:
        raise RuntimeError("one_scan has no 'obj_pcds'.")

    xyz_parts, rgb_parts, nrm_parts = [], [], []
    have_rgb = True
    have_normals = True

    for i, obj in enumerate(obj_pcds):
        arr = np.asarray(obj)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise RuntimeError(f"obj_pcds[{i}] invalid shape: {arr.shape}")

        xyz = arr[:, :3].astype(np.float32, copy=False)
        xyz_parts.append(xyz)

        if arr.shape[1] >= 6:
            rgb = arr[:, 3:6].astype(np.float32, copy=False)
            # In your loader, colors are already normalized to [-1, 1].
            # Open3D expects [0,1], so remap if needed.
            if rgb.min() < 0.0:
                rgb01 = np.clip((rgb + 1.0) * 0.5, 0.0, 1.0)
            elif rgb.max() > 1.5:
                rgb01 = np.clip(rgb / 255.0, 0.0, 1.0)
            else:
                rgb01 = np.clip(rgb, 0.0, 1.0)
            rgb_parts.append(rgb01)
        else:
            have_rgb = False

        if arr.shape[1] >= 9:
            nrm = arr[:, 6:9].astype(np.float32, copy=False)
            nrm_parts.append(nrm)
        else:
            have_normals = False

    xyz_cat = np.concatenate(xyz_parts, axis=0)
    rgb01_cat = np.concatenate(rgb_parts, axis=0) if (have_rgb and len(rgb_parts) == len(obj_pcds)) else None
    normals_cat = np.concatenate(nrm_parts, axis=0) if (have_normals and len(nrm_parts) == len(obj_pcds)) else None

    if require_normals and normals_cat is None:
        raise RuntimeError(
            "Normals not found in obj_pcds. Expected Nx9 per object (xyzrgb + normals). "
            "Check that your _load_one_scan() is concatenating normals into obj_pcds."
        )

    return xyz_cat, rgb01_cat, normals_cat


def visualize_scene(
    xyz: np.ndarray,
    rgb01: np.ndarray | None,
    normals: np.ndarray | None,
    *,
    voxel: float = 0.0,
    point_size: float = 2.0,
    show_normals: bool = True,
    color_by_normals: bool = True,
    keep_scene_rgb: bool = False,
    orient_viewpoint: np.ndarray | None = None,
    window_name: str = "Scene Normals Viewer (Loader)",
):
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    n_vis = None
    if normals is not None:
        n_vis = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        if n_vis.shape[0] != xyz.shape[0]:
            raise ValueError(f"Points/normals mismatch: points={xyz.shape[0]} normals={n_vis.shape[0]}")
        n_vis = normalize_normals(n_vis)

        if orient_viewpoint is not None:
            n_vis = orient_normals_toward_viewpoint(xyz, n_vis, orient_viewpoint)

        pcd.normals = o3d.utility.Vector3dVector(n_vis)

    # Colors
    if color_by_normals and (n_vis is not None) and (not keep_scene_rgb):
        pcd.colors = o3d.utility.Vector3dVector(normals_to_rgb01(n_vis))
    elif rgb01 is not None and np.asarray(rgb01).shape[0] == xyz.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb01, 0.0, 1.0))
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)

    ro = vis.get_render_option()
    ro.point_size = float(point_size)
    ro.point_show_normal = bool(show_normals and (n_vis is not None))

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Visualize dense ScanNet scene normals from _load_one_scan().")
    parser.add_argument("--cfg", default="msr3d/configs/data.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--scan_id", default="scene0000_00")
    parser.add_argument("--normals_dir", default=None, help="(Not used in this version)")
    parser.add_argument("--voxel", type=float, default=0.0)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--no_normals", action="store_true", help="Do not draw normal glyphs.")
    parser.add_argument("--color_by_normals", action="store_true", help="Color points by normal direction RGB=(n+1)/2.")
    parser.add_argument("--keep_scene_rgb", action="store_true", help="Do not override RGB when coloring by normals.")
    parser.add_argument("--normal_colors", action="store_true", help="Color by normal direction but do not draw normal glyphs.")
    parser.add_argument(
        "--orient_viewpoint",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional viewpoint (x y z) to flip normals toward.",
    )

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    loader = ScanNetBase(cfg, split=args.split)

    # IMPORTANT: This must be the version of _load_one_scan that loads normals and concatenates them into obj_pcds.
    _, one_scan = loader._load_one_scan(args.scan_id, load_inst_info=True, load_pc_info=True)

    xyz_scene, rgb01_scene, normals_scene = load_scene_from_loader(one_scan, require_normals=True)

    print(f"[info] scan_id={args.scan_id}")
    print(f"[info] scene points={xyz_scene.shape[0]} scene normals={normals_scene.shape[0]}")
    print(f"[info] color_by_normals={args.color_by_normals} keep_scene_rgb={args.keep_scene_rgb}")

    visualize_scene(
        xyz=xyz_scene,
        rgb01=rgb01_scene,
        normals=None if (args.no_normals and not args.color_by_normals) else normals_scene,
        voxel=args.voxel,
        point_size=args.point_size,
        show_normals=(not args.no_normals),
        color_by_normals=args.color_by_normals,
        keep_scene_rgb=args.keep_scene_rgb,
        orient_viewpoint=None if args.orient_viewpoint is None else np.array(args.orient_viewpoint, dtype=np.float32),
        window_name="Normals from _load_one_scan",
    )


if __name__ == "__main__":
    main()
