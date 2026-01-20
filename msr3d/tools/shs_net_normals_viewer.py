
# #!/usr/bin/env python3
# """
# SHS-Net Normals Viewer (robust)

# Problem this fixes:
# - MSR3D/ScanNetBase can build a scene point cloud by concatenating obj_pcds (millions of points),
#   while SHS-Net predicts normals for a different (usually downsampled) point set (e.g., ~81k).
# - This script will NOT crash on mismatch. It will choose a safe fallback path.

# Modes:
# 1) --mode pth
#    Visualize SHS-Net normals on the SAME point cloud SHS-Net used (load from --pc_pth).  (Recommended)

# 2) --mode scene_nn
#    Visualize on the dense scene cloud (concatenated obj_pcds) by projecting SHS-Net normals to scene points
#    using nearest neighbor mapping. Requires --pc_pth (the SHS-Net input points) to build the NN map.

# 3) --mode scene
#    Visualize the dense scene cloud only. If normals mismatch, it will show points only (no crash).

# Usage examples:
# - Recommended:
#   python shs_net_normals_viewer_fixed.py --scan_id scene0000_00 --mode pth \
#     --pc_pth /path/to/scene0000_00.pth \
#     --normals_dir /path/to/pred_normal

# - Project to dense scene:
#   python shs_net_normals_viewer_fixed.py --scan_id scene0000_00 --mode scene_nn \
#     --pc_pth /path/to/scene0000_00.pth \
#     --normals_dir /path/to/pred_normal --voxel 0.02

# Notes:
# - If your SHS-Net saves .npy normals, pass --normals_path to that file.
# - If your SHS-Net saves .normals text, the script loads it.
# """

# import os
# import argparse
# import numpy as np
# import open3d as o3d
# import torch
# from scipy.spatial import cKDTree
# from omegaconf import OmegaConf

# # Your project import
# from data.datasets.scannet_base import ScanNetBase


# def load_normals(normals_path: str) -> np.ndarray:
#     """
#     Load normals from:
#       - .npy: (N,3)
#       - text (.normals/.txt): whitespace-separated rows: nx ny nz
#     Returns: (N,3) float32
#     """
#     if not os.path.exists(normals_path):
#         raise FileNotFoundError(f"Normals file not found: {normals_path}")

#     ext = os.path.splitext(normals_path)[1].lower()
#     if ext == ".npy":
#         arr = np.load(normals_path)
#         arr = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
#         if arr.shape[1] != 3:
#             raise ValueError(f"{normals_path}: expected Nx3, got {arr.shape}")
#         return arr

#     # Text loader
#     normals = []
#     with open(normals_path, "r") as f:
#         for line_no, line in enumerate(f, start=1):
#             s = line.strip()
#             if not s or s.startswith("#"):
#                 continue
#             s = s.replace(",", " ")
#             parts = s.split()
#             if len(parts) < 3:
#                 raise ValueError(f"{normals_path}:{line_no}: expected 3 floats, got: '{line.strip()}'")
#             try:
#                 nx, ny, nz = float(parts[0]), float(parts[1]), float(parts[2])
#             except ValueError as e:
#                 raise ValueError(f"{normals_path}:{line_no}: could not parse floats: '{line.strip()}'") from e
#             normals.append([nx, ny, nz])

#     if len(normals) == 0:
#         raise ValueError(f"No normals loaded from: {normals_path}")

#     return np.asarray(normals, dtype=np.float32)


# def resolve_normals_path(normals_dir: str, scan_id: str, normals_path: str | None) -> str:
#     """
#     Priority:
#       1) explicit --normals_path
#       2) <normals_dir>/<scan_id>.normals
#       3) <normals_dir>/<scan_id>.npy
#       4) <normals_dir>/<scan_id>.txt
#       5) <normals_dir>/<scan_id>_normal.npy
#     """
#     if normals_path is not None:
#         return normals_path

#     cands = [
#         os.path.join(normals_dir, f"{scan_id}.normals"),
#         os.path.join(normals_dir, f"{scan_id}.npy"),
#         os.path.join(normals_dir, f"{scan_id}.txt"),
#         os.path.join(normals_dir, f"{scan_id}_normal.npy"),
#     ]
#     for p in cands:
#         if os.path.exists(p):
#             return p

#     raise FileNotFoundError(
#         "Could not resolve normals file. Tried:\n"
#         + "\n".join(cands)
#         + "\nProvide --normals_path explicitly or ensure the file naming matches."
#     )


# def load_pth_xyz(pth_path: str) -> np.ndarray:
#     """
#     Load point cloud xyz from a .pth file.

#     Supported payloads:
#       - tuple/list where element 0 is Nx3 (or Nx>=3): uses [:, :3]
#       - torch.Tensor Nx3
#       - numpy.ndarray Nx3

#     Returns: (N,3) float32
#     """
#     if not os.path.exists(pth_path):
#         raise FileNotFoundError(f"Point cloud .pth not found: {pth_path}")

#     obj = torch.load(pth_path, map_location="cpu", weights_only=False)

#     if isinstance(obj, (tuple, list)) and len(obj) >= 1:
#         xyz = np.asarray(obj[0], dtype=np.float32)
#     elif isinstance(obj, torch.Tensor):
#         xyz = obj.detach().cpu().numpy().astype(np.float32, copy=False)
#     elif isinstance(obj, np.ndarray):
#         xyz = obj.astype(np.float32, copy=False)
#     else:
#         raise TypeError(f"Unsupported .pth payload type in {pth_path}: {type(obj)}")

#     if xyz.ndim != 2 or xyz.shape[1] < 3:
#         raise ValueError(f"{pth_path}: expected Nx3 (or Nx>=3), got {xyz.shape}")

#     return xyz[:, :3].astype(np.float32, copy=False)


# def load_scene_points(one_scan: dict) -> tuple[np.ndarray, np.ndarray | None, int]:
#     """
#     Build dense scene point cloud by concatenating per-entry arrays in one_scan["obj_pcds"].
#     Returns:
#       xyz: (N,3) float32
#       rgb01: (N,3) float32 in [0,1] or None
#       obj_entries: int, number of entries concatenated (not necessarily semantic objects)
#     """
#     obj_pcds = one_scan.get("obj_pcds", None)
#     if obj_pcds is None or len(obj_pcds) == 0:
#         raise RuntimeError("one_scan has no 'obj_pcds' to build a scene point cloud.")

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

#     return xyz, rgb01, len(obj_pcds)


# def normalize_normals(normals: np.ndarray) -> np.ndarray:
#     normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
#     nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
#     return normals / nlen


# def visualize_open3d(
#     xyz: np.ndarray,
#     rgb01: np.ndarray | None,
#     normals: np.ndarray | None,
#     voxel: float = 0.0,
#     point_size: float = 2.0,
#     show_normals: bool = True,
#     window_name: str = "Scene Normals Viewer",
# ):
#     xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)

#     if rgb01 is not None and np.asarray(rgb01).shape[0] == xyz.shape[0]:
#         pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb01, 0.0, 1.0))
#     else:
#         pcd.paint_uniform_color([0.7, 0.7, 0.7])

#     if normals is not None:
#         normals = normalize_normals(normals)
#         if normals.shape[0] != xyz.shape[0]:
#             raise ValueError(f"Internal error: xyz={xyz.shape[0]} normals={normals.shape[0]}")
#         pcd.normals = o3d.utility.Vector3dVector(normals)

#     if voxel and voxel > 0:
#         pcd = pcd.voxel_down_sample(voxel_size=float(voxel))

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=window_name)
#     vis.add_geometry(pcd)

#     ro = vis.get_render_option()
#     ro.point_size = float(point_size)
#     ro.point_show_normal = bool(show_normals and (normals is not None))

#     vis.run()
#     vis.destroy_window()


# def project_normals_nn(xyz_src: np.ndarray, n_src: np.ndarray, xyz_tgt: np.ndarray) -> np.ndarray:
#     """
#     Nearest-neighbor project normals from source points to target points.
#     """
#     xyz_src = np.asarray(xyz_src, dtype=np.float32).reshape(-1, 3)
#     xyz_tgt = np.asarray(xyz_tgt, dtype=np.float32).reshape(-1, 3)
#     n_src = np.asarray(n_src, dtype=np.float32).reshape(-1, 3)

#     if xyz_src.shape[0] != n_src.shape[0]:
#         raise ValueError(f"Source points/normals mismatch: src_points={xyz_src.shape[0]} src_normals={n_src.shape[0]}")

#     tree = cKDTree(xyz_src)
#     _, nn = tree.query(xyz_tgt, k=1, workers=-1)
#     return n_src[nn]


# def main():
#     parser = argparse.ArgumentParser(description="Robust SHS-Net normals viewer.")
#     parser.add_argument("--cfg", default="msr3d/configs/data.yaml", help="Path to msr3d data.yaml")
#     parser.add_argument("--split", default="train", help="Dataset split (train/val/test)")
#     parser.add_argument("--scan_id", default="scene0000_00", help="ScanNet scene id, e.g. scene0000_00")

#     parser.add_argument(
#         "--normals_dir",
#         default="",
#         help="Directory containing predicted normals files.",
#     )
#     parser.add_argument(
#         "--normals_path",
#         default=None,
#         help="Explicit normals file path (.normals/.txt/.npy). Overrides --normals_dir resolution.",
#     )

#     parser.add_argument(
#         "--pc_pth",
#         default=None,
#         help="Path to the SHS-Net input point cloud .pth for this scan_id (needed for --mode pth or scene_nn).",
#     )

#     parser.add_argument(
#         "--mode",
#         choices=["pth", "scene", "scene_nn"],
#         default="pth",
#         help=(
#             "pth: visualize on SHS-Net input points (requires --pc_pth). "
#             "scene: visualize dense scene points; if mismatch, show points only. "
#             "scene_nn: visualize dense scene points with NN-projected normals (requires --pc_pth)."
#         ),
#     )

#     parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (0 disables)")
#     parser.add_argument("--point_size", type=float, default=2.0, help="Open3D point size")
#     parser.add_argument("--no_normals", action="store_true", help="Do not show normals (points only)")
#     args = parser.parse_args()

#     # Load normals
#     if args.normals_path is None and not args.normals_dir:
#         raise ValueError("Provide either --normals_path or --normals_dir.")

#     normals_path = resolve_normals_path(args.normals_dir, args.scan_id, args.normals_path)
#     normals = load_normals(normals_path)

#     print(f"[info] scan_id={args.scan_id}")
#     print(f"[info] normals_path={normals_path} normals={normals.shape[0]}")

#     # Mode: pth (recommended)
#     if args.mode == "pth":
#         if args.pc_pth is None:
#             raise ValueError("--mode pth requires --pc_pth (the SHS-Net input .pth for this scan).")
#         xyz = load_pth_xyz(args.pc_pth)
#         rgb01 = None
#         print(f"[info] pc_pth={args.pc_pth} points={xyz.shape[0]}")

#         if xyz.shape[0] != normals.shape[0]:
#             raise ValueError(
#                 f"[error] SHS-Net input points and normals mismatch. points={xyz.shape[0]} normals={normals.shape[0]}\n"
#                 f"Make sure --pc_pth is the exact point set used to produce these normals."
#             )

#         visualize_open3d(
#             xyz=xyz,
#             rgb01=rgb01,
#             normals=None if args.no_normals else normals,
#             voxel=args.voxel,
#             point_size=args.point_size,
#             show_normals=(not args.no_normals),
#             window_name="SHS-Net Normals (PTH points)",
#         )
#         return

#     # Load dense scene points (obj_pcds concatenation)
#     cfg = OmegaConf.load(args.cfg)
#     loader = ScanNetBase(cfg, split=args.split)
#     _, one_scan = loader._load_one_scan(args.scan_id, load_inst_info=True, load_pc_info=True)

#     # Optional debugging: counts of instance-like fields (won't break if missing)
#     for k in ["obj_pcds", "inst_ids", "instances", "inst_info", "obj_ids", "gt_inst_ids"]:
#         if k in one_scan:
#             try:
#                 print(f"[info] one_scan[{k}] len={len(one_scan[k])}")
#             except Exception:
#                 print(f"[info] one_scan[{k}] type={type(one_scan[k])}")

#     xyz_scene, rgb01_scene, obj_entries = load_scene_points(one_scan)
#     print(f"[info] built scene from obj_pcds entries={obj_entries} points={xyz_scene.shape[0]}")

#     # Mode: scene (points only if mismatch)
#     if args.mode == "scene":
#         if xyz_scene.shape[0] != normals.shape[0]:
#             print(
#                 f"[warn] Points/normals mismatch (scene_points={xyz_scene.shape[0]} normals={normals.shape[0]}). "
#                 f"Showing points only. Use --mode scene_nn (with --pc_pth) to project normals, "
#                 f"or --mode pth to view on SHS-Net points."
#             )
#             normals_vis = None
#         else:
#             normals_vis = None if args.no_normals else normals

#         visualize_open3d(
#             xyz=xyz_scene,
#             rgb01=rgb01_scene,
#             normals=normals_vis,
#             voxel=args.voxel,
#             point_size=args.point_size,
#             show_normals=(not args.no_normals),
#             window_name="Dense Scene Viewer",
#         )
#         return

#     # Mode: scene_nn (project normals to dense scene)
#     if args.mode == "scene_nn":
#         if args.pc_pth is None:
#             raise ValueError("--mode scene_nn requires --pc_pth (SHS-Net input points) to build the NN projection.")
#         xyz_src = load_pth_xyz(args.pc_pth)
#         print(f"[info] pc_pth={args.pc_pth} points={xyz_src.shape[0]}")

#         if xyz_src.shape[0] != normals.shape[0]:
#             raise ValueError(
#                 f"[error] SHS-Net input points and normals mismatch. src_points={xyz_src.shape[0]} normals={normals.shape[0]}\n"
#                 f"Make sure --pc_pth is the exact point set used to produce these normals."
#             )

#         if args.no_normals:
#             normals_scene = None
#         else:
#             print("[info] projecting normals to dense scene via nearest neighbor (this may take a bit for millions of points)...")
#             normals_scene = project_normals_nn(xyz_src=xyz_src, n_src=normals, xyz_tgt=xyz_scene)
#             print(f"[info] projected normals: {normals_scene.shape}")

#         visualize_open3d(
#             xyz=xyz_scene,
#             rgb01=rgb01_scene,
#             normals=normals_scene,
#             voxel=args.voxel,
#             point_size=args.point_size,
#             show_normals=(not args.no_normals),
#             window_name="Dense Scene + NN-projected SHS-Net Normals",
#         )
#         return


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
SHS-Net Normals Viewer (robust) + Normal-direction color indexing

Adds:
- --color_by_normals: color points by normal direction using RGB = (n + 1) / 2
- --keep_scene_rgb: when enabled, do NOT override scene RGB even if --color_by_normals is set

Notes:
- Coloring by normals is independent from showing normal glyphs (Open3D's point_show_normal).
- Works in all modes (pth / scene / scene_nn). If normals are not available, falls back to scene RGB or gray.

Usage examples:
  python shs_net_normals_viewer_fixed.py --scan_id scene0000_00 --mode pth \
    --pc_pth /path/to/scene0000_00.pth \
    --normals_dir /path/to/pred_normal \
    --color_by_normals

  python shs_net_normals_viewer_fixed.py --scan_id scene0000_00 --mode scene_nn \
    --pc_pth /path/to/scene0000_00.pth \
    --normals_dir /path/to/pred_normal --voxel 0.02 \
    --color_by_normals
"""

import os
import argparse
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree
from omegaconf import OmegaConf

# Your project import
from data.datasets.scannet_base import ScanNetBase


def load_normals(normals_path: str) -> np.ndarray:
    """
    Load normals from:
      - .npy: (N,3)
      - text (.normals/.txt): whitespace-separated rows: nx ny nz
    Returns: (N,3) float32
    """
    if not os.path.exists(normals_path):
        raise FileNotFoundError(f"Normals file not found: {normals_path}")

    ext = os.path.splitext(normals_path)[1].lower()
    if ext == ".npy":
        arr = np.load(normals_path)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, 3)
        if arr.shape[1] != 3:
            raise ValueError(f"{normals_path}: expected Nx3, got {arr.shape}")
        return arr

    # Text loader
    normals = []
    with open(normals_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            parts = s.split()
            if len(parts) < 3:
                raise ValueError(f"{normals_path}:{line_no}: expected 3 floats, got: '{line.strip()}'")
            try:
                nx, ny, nz = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError as e:
                raise ValueError(f"{normals_path}:{line_no}: could not parse floats: '{line.strip()}'") from e
            normals.append([nx, ny, nz])

    if len(normals) == 0:
        raise ValueError(f"No normals loaded from: {normals_path}")

    return np.asarray(normals, dtype=np.float32)


def resolve_normals_path(normals_dir: str, scan_id: str, normals_path: str | None) -> str:
    """
    Priority:
      1) explicit --normals_path
      2) <normals_dir>/<scan_id>.normals
      3) <normals_dir>/<scan_id>.npy
      4) <normals_dir>/<scan_id>.txt
      5) <normals_dir>/<scan_id>_normal.npy
    """
    if normals_path is not None:
        return normals_path

    cands = [
        os.path.join(normals_dir, f"{scan_id}.normals"),
        os.path.join(normals_dir, f"{scan_id}.npy"),
        os.path.join(normals_dir, f"{scan_id}.txt"),
        os.path.join(normals_dir, f"{scan_id}_normal.npy"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Could not resolve normals file. Tried:\n"
        + "\n".join(cands)
        + "\nProvide --normals_path explicitly or ensure the file naming matches."
    )


def load_pth_xyz(pth_path: str) -> np.ndarray:
    """
    Load point cloud xyz from a .pth file.

    Supported payloads:
      - tuple/list where element 0 is Nx3 (or Nx>=3): uses [:, :3]
      - torch.Tensor Nx3
      - numpy.ndarray Nx3

    Returns: (N,3) float32
    """
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Point cloud .pth not found: {pth_path}")

    obj = torch.load(pth_path, map_location="cpu", weights_only=False)

    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        xyz = np.asarray(obj[0], dtype=np.float32)
    elif isinstance(obj, torch.Tensor):
        xyz = obj.detach().cpu().numpy().astype(np.float32, copy=False)
    elif isinstance(obj, np.ndarray):
        xyz = obj.astype(np.float32, copy=False)
    else:
        raise TypeError(f"Unsupported .pth payload type in {pth_path}: {type(obj)}")

    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"{pth_path}: expected Nx3 (or Nx>=3), got {xyz.shape}")

    return xyz[:, :3].astype(np.float32, copy=False)


def load_scene_points(one_scan: dict) -> tuple[np.ndarray, np.ndarray | None, int]:
    """
    Build dense scene point cloud by concatenating per-entry arrays in one_scan["obj_pcds"].
    Returns:
      xyz: (N,3) float32
      rgb01: (N,3) float32 in [0,1] or None
      obj_entries: int, number of entries concatenated (not necessarily semantic objects)
    """
    obj_pcds = one_scan.get("obj_pcds", None)
    if obj_pcds is None or len(obj_pcds) == 0:
        raise RuntimeError("one_scan has no 'obj_pcds' to build a scene point cloud.")

    parts = []
    for i, obj in enumerate(obj_pcds):
        arr = np.asarray(obj)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise RuntimeError(f"obj_pcds[{i}] has invalid shape: {arr.shape}")
        parts.append(arr)

    scene = np.concatenate(parts, axis=0)

    xyz = scene[:, :3].astype(np.float32)

    rgb01 = None
    if scene.shape[1] >= 6:
        rgb = scene[:, 3:6].astype(np.float32)
        if rgb.size > 0 and rgb.max() > 1.5:
            rgb01 = np.clip(rgb / 255.0, 0.0, 1.0)
        else:
            rgb01 = np.clip(rgb, 0.0, 1.0)

    return xyz, rgb01, len(obj_pcds)


def normalize_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals / nlen


def normals_to_rgb01(normals: np.ndarray) -> np.ndarray:
    """
    Map unit normals in [-1,1] to RGB in [0,1] via RGB = (n + 1) / 2.
    """
    n = normalize_normals(normals)
    rgb = 0.5 * (n + 1.0)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def visualize_open3d(
    xyz: np.ndarray,
    rgb01: np.ndarray | None,
    normals: np.ndarray | None,
    voxel: float = 0.0,
    point_size: float = 2.0,
    show_normals: bool = True,
    window_name: str = "Scene Normals Viewer",
    color_by_normals: bool = False,
    keep_scene_rgb: bool = False,
):
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Attach normals first (so we can color by them if requested)
    if normals is not None:
        normals = normalize_normals(normals)
        if normals.shape[0] != xyz.shape[0]:
            raise ValueError(f"Internal error: xyz={xyz.shape[0]} normals={normals.shape[0]}")
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # Choose colors
    if color_by_normals and (normals is not None) and (not keep_scene_rgb):
        pcd.colors = o3d.utility.Vector3dVector(normals_to_rgb01(normals))
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
    ro.point_show_normal = bool(show_normals and (normals is not None))

    vis.run()
    vis.destroy_window()


def project_normals_nn(xyz_src: np.ndarray, n_src: np.ndarray, xyz_tgt: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbor project normals from source points to target points.
    """
    xyz_src = np.asarray(xyz_src, dtype=np.float32).reshape(-1, 3)
    xyz_tgt = np.asarray(xyz_tgt, dtype=np.float32).reshape(-1, 3)
    n_src = np.asarray(n_src, dtype=np.float32).reshape(-1, 3)

    if xyz_src.shape[0] != n_src.shape[0]:
        raise ValueError(f"Source points/normals mismatch: src_points={xyz_src.shape[0]} src_normals={n_src.shape[0]}")

    tree = cKDTree(xyz_src)
    _, nn = tree.query(xyz_tgt, k=1, workers=-1)
    return n_src[nn]


def main():
    parser = argparse.ArgumentParser(description="Robust SHS-Net normals viewer with optional normal-direction coloring.")
    parser.add_argument("--cfg", default="msr3d/configs/data.yaml", help="Path to msr3d data.yaml")
    parser.add_argument("--split", default="train", help="Dataset split (train/val/test)")
    parser.add_argument("--scan_id", default="scene0000_00", help="ScanNet scene id, e.g. scene0000_00")

    parser.add_argument(
        "--normals_dir",
        default="",
        help="Directory containing predicted normals files.",
    )
    parser.add_argument(
        "--normals_path",
        default=None,
        help="Explicit normals file path (.normals/.txt/.npy). Overrides --normals_dir resolution.",
    )

    parser.add_argument(
        "--pc_pth",
        default=None,
        help="Path to the SHS-Net input point cloud .pth for this scan_id (needed for --mode pth or scene_nn).",
    )

    parser.add_argument(
        "--mode",
        choices=["pth", "scene", "scene_nn"],
        default="pth",
        help=(
            "pth: visualize on SHS-Net input points (requires --pc_pth). "
            "scene: visualize dense scene points; if mismatch, show points only. "
            "scene_nn: visualize dense scene points with NN-projected normals (requires --pc_pth)."
        ),
    )

    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (0 disables)")
    parser.add_argument("--point_size", type=float, default=2.0, help="Open3D point size")
    parser.add_argument("--no_normals", action="store_true", help="Do not show normals (points only)")

    # NEW: normal-direction color indexing
    parser.add_argument(
        "--color_by_normals",
        action="store_true",
        help="Color points by normal direction using RGB=(n+1)/2. Overrides scene RGB if available.",
    )
    parser.add_argument(
        "--keep_scene_rgb",
        action="store_true",
        help="If set, do NOT override scene RGB even when --color_by_normals is enabled.",
    )

    args = parser.parse_args()

    # Load normals
    if args.normals_path is None and not args.normals_dir:
        raise ValueError("Provide either --normals_path or --normals_dir.")

    normals_path = resolve_normals_path(args.normals_dir, args.scan_id, args.normals_path)
    normals = load_normals(normals_path)

    print(f"[info] scan_id={args.scan_id}")
    print(f"[info] normals_path={normals_path} normals={normals.shape[0]}")

    # Mode: pth (recommended)
    if args.mode == "pth":
        if args.pc_pth is None:
            raise ValueError("--mode pth requires --pc_pth (the SHS-Net input .pth for this scan).")
        xyz = load_pth_xyz(args.pc_pth)
        rgb01 = None
        print(f"[info] pc_pth={args.pc_pth} points={xyz.shape[0]}")

        if xyz.shape[0] != normals.shape[0]:
            raise ValueError(
                f"[error] SHS-Net input points and normals mismatch. points={xyz.shape[0]} normals={normals.shape[0]}\n"
                f"Make sure --pc_pth is the exact point set used to produce these normals."
            )

        visualize_open3d(
            xyz=xyz,
            rgb01=rgb01,
            normals=None if args.no_normals else normals,
            voxel=args.voxel,
            point_size=args.point_size,
            show_normals=(not args.no_normals),
            window_name="SHS-Net Normals (PTH points)",
            color_by_normals=args.color_by_normals,
            keep_scene_rgb=args.keep_scene_rgb,
        )
        return

    # Load dense scene points (obj_pcds concatenation)
    cfg = OmegaConf.load(args.cfg)
    loader = ScanNetBase(cfg, split=args.split)
    _, one_scan = loader._load_one_scan(args.scan_id, load_inst_info=True, load_pc_info=True)

    # Optional debugging: counts of instance-like fields (won't break if missing)
    for k in ["obj_pcds", "inst_ids", "instances", "inst_info", "obj_ids", "gt_inst_ids"]:
        if k in one_scan:
            try:
                print(f"[info] one_scan[{k}] len={len(one_scan[k])}")
            except Exception:
                print(f"[info] one_scan[{k}] type={type(one_scan[k])}")

    xyz_scene, rgb01_scene, obj_entries = load_scene_points(one_scan)
    print(f"[info] built scene from obj_pcds entries={obj_entries} points={xyz_scene.shape[0]}")

    # Mode: scene (points only if mismatch)
    if args.mode == "scene":
        if xyz_scene.shape[0] != normals.shape[0]:
            print(
                f"[warn] Points/normals mismatch (scene_points={xyz_scene.shape[0]} normals={normals.shape[0]}). "
                f"Showing points only. Use --mode scene_nn (with --pc_pth) to project normals, "
                f"or --mode pth to view on SHS-Net points."
            )
            normals_vis = None
        else:
            normals_vis = None if args.no_normals else normals

        visualize_open3d(
            xyz=xyz_scene,
            rgb01=rgb01_scene,
            normals=normals_vis,
            voxel=args.voxel,
            point_size=args.point_size,
            show_normals=(not args.no_normals),
            window_name="Dense Scene Viewer",
            color_by_normals=args.color_by_normals,
            keep_scene_rgb=args.keep_scene_rgb,
        )
        return

    # Mode: scene_nn (project normals to dense scene)
    if args.mode == "scene_nn":
        if args.pc_pth is None:
            raise ValueError("--mode scene_nn requires --pc_pth (SHS-Net input points) to build the NN projection.")
        xyz_src = load_pth_xyz(args.pc_pth)
        print(f"[info] pc_pth={args.pc_pth} points={xyz_src.shape[0]}")

        if xyz_src.shape[0] != normals.shape[0]:
            raise ValueError(
                f"[error] SHS-Net input points and normals mismatch. src_points={xyz_src.shape[0]} normals={normals.shape[0]}\n"
                f"Make sure --pc_pth is the exact point set used to produce these normals."
            )

        if args.no_normals:
            normals_scene = None
        else:
            print("[info] projecting normals to dense scene via nearest neighbor (this may take a bit for millions of points)...")
            normals_scene = project_normals_nn(xyz_src=xyz_src, n_src=normals, xyz_tgt=xyz_scene)
            print(f"[info] projected normals: {normals_scene.shape}")

        visualize_open3d(
            xyz=xyz_scene,
            rgb01=rgb01_scene,
            normals=normals_scene,
            voxel=args.voxel,
            point_size=args.point_size,
            show_normals=(not args.no_normals),
            window_name="Dense Scene + NN-projected SHS-Net Normals",
            color_by_normals=args.color_by_normals,
            keep_scene_rgb=args.keep_scene_rgb,
        )
        return


if __name__ == "__main__":
    main()
