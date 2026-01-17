import os
import argparse
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf
from data.datasets.scannet_base import ScanNetBase


def load_normals_txt(normals_path: str) -> np.ndarray:
    """
    Load normals from a text file with rows: nx ny nz
    - whitespace separated floats
    - optional commas
    - ignores empty lines and comment lines starting with '#'
    Returns: (N,3) float32
    """
    if not os.path.exists(normals_path):
        raise FileNotFoundError(f"Normals file not found: {normals_path}")

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
      3) <normals_dir>/<scan_id>.txt
      4) <normals_dir>/<scan_id>_normals.txt
    """
    if normals_path is not None:
        return normals_path

    cands = [
        os.path.join(normals_dir, f"{scan_id}.normals"),
        os.path.join(normals_dir, f"{scan_id}.txt"),
        os.path.join(normals_dir, f"{scan_id}_normals.txt"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Could not resolve normals file. Tried:\n"
        + "\n".join(cands)
        + "\nProvide --normals_path explicitly or ensure the file naming matches."
    )


def load_scene_points(one_scan: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Build a scene-level point cloud by concatenating per-object point clouds in one_scan["obj_pcds"].

    Expects obj_pcds: list of (Ni, 6) arrays [x y z r g b] (RGB typically 0-255).
    Returns:
      xyz: (N,3) float32
      rgb01: (N,3) float32 in [0,1] or None
    """
    obj_pcds = one_scan.get("obj_pcds", None)
    if obj_pcds is None or len(obj_pcds) == 0:
        raise RuntimeError("one_scan has no 'obj_pcds' to build a scene point cloud.")

    # Concatenate all objects
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
        # handle 0-255 vs 0-1
        if rgb.max() > 1.5:
            rgb01 = np.clip(rgb / 255.0, 0.0, 1.0)
        else:
            rgb01 = np.clip(rgb, 0.0, 1.0)

    print(f"[info] built scene by concatenating obj_pcds: objects={len(obj_pcds)} points={xyz.shape[0]}")
    return xyz, rgb01


def visualize_scene_with_normals(
    xyz: np.ndarray,
    rgb01: np.ndarray | None,
    normals: np.ndarray,
    voxel: float = 0.0,
    point_size: float = 2.0,
    show_normals: bool = True,
):
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)

    if xyz.shape[0] != normals.shape[0]:
        raise ValueError(f"Points/normals mismatch: points={xyz.shape[0]} normals={normals.shape[0]}")

    # Normalize normals (defensive)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals = normals / nlen

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    if rgb01 is not None and rgb01.shape[0] == xyz.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb01, 0.0, 1.0))
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # Optional voxel downsample for speed
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Scene Normals Viewer (.normals)")
    vis.add_geometry(pcd)

    ro = vis.get_render_option()
    ro.point_size = float(point_size)
    ro.point_show_normal = bool(show_normals)

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Visualize scene-level normals from a text .normals file.")
    parser.add_argument("--cfg", default="msr3d/configs/data.yaml", help="Path to msr3d data.yaml")
    parser.add_argument("--split", default="train", help="Dataset split (train/val/test)")
    parser.add_argument("--scan_id", default="scene0000_00", help="ScanNet scene id, e.g. scene0000_00")

    parser.add_argument(
        "--normals_dir",
        default="/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/scene_normals_txt/",
        help="Directory containing scene .normals files",
    )
    parser.add_argument(
        "--normals_path",
        default=None,
        help="Explicit path to a .normals file (overrides --normals_dir naming rules)",
    )

    parser.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size (0 disables)")
    parser.add_argument("--point_size", type=float, default=2.0, help="Open3D point size")
    parser.add_argument("--no_normals", action="store_true", help="Do not show normals (points only)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    loader = ScanNetBase(cfg, split=args.split)

    _, one_scan = loader._load_one_scan(args.scan_id, load_inst_info=True, load_pc_info=True)

    xyz, rgb01 = load_scene_points(one_scan)

    normals_path = resolve_normals_path(args.normals_dir, args.scan_id, args.normals_path)
    normals = load_normals_txt(normals_path)

    print(f"[info] scan_id={args.scan_id} points={xyz.shape[0]}")
    print(f"[info] normals_path={normals_path} normals={normals.shape[0]}")

    visualize_scene_with_normals(
        xyz=xyz,
        rgb01=rgb01,
        normals=normals,
        voxel=args.voxel,
        point_size=args.point_size,
        show_normals=(not args.no_normals),
    )


if __name__ == "__main__":
    main()
