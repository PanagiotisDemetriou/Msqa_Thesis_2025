# import argparse
# import numpy as np
# import open3d as o3d
# from omegaconf import OmegaConf
# from data.datasets.scannet_base import ScanNetBase


# def normalize_normals(normals: np.ndarray) -> np.ndarray:
#     normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
#     nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
#     return normals / nlen


# def normals_to_rgb01(normals: np.ndarray) -> np.ndarray:
#     n = normalize_normals(normals)
#     rgb = 0.5 * (n + 1.0)
#     return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


# def orient_normals_toward_viewpoint(xyz: np.ndarray, normals: np.ndarray, viewpoint: np.ndarray) -> np.ndarray:
#     xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
#     n = normalize_normals(normals)
#     vp = np.asarray(viewpoint, dtype=np.float32).reshape(1, 3)
#     v = vp - xyz
#     flip = (np.sum(n * v, axis=1) < 0.0).astype(np.float32).reshape(-1, 1)
#     return n * (1.0 - 2.0 * flip)


# def load_scene_from_scene_fts(one_scan: dict):
#     """
#     one_scan["scene_fts"] expected shape: (N,9) = [xyz, rgb(-1..1), normals]
#     Returns:
#       xyz (N,3), rgb01 (N,3), normals (N,3)
#     """
#     if "scene_fts" not in one_scan:
#         raise RuntimeError("one_scan is missing key 'scene_fts'. "
#                            "Your _load_one_scan must set one_scan['scene_fts']=(N,9).")

#     scene = np.asarray(one_scan["scene_fts"], dtype=np.float32)
#     if scene.ndim != 2 or scene.shape[1] < 9:
#         raise ValueError(f"scene_fts has invalid shape {scene.shape}; expected (N,9).")

#     xyz = scene[:, 0:3].astype(np.float32, copy=False)
#     rgb_m11 = scene[:, 3:6].astype(np.float32, copy=False)   # [-1,1]
#     normals = scene[:, 6:9].astype(np.float32, copy=False)

#     # Open3D wants colors in [0,1]
#     rgb01 = np.clip((rgb_m11 + 1.0) * 0.5, 0.0, 1.0).astype(np.float32, copy=False)

#     return xyz, rgb01, normals


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
#     window_name: str = "Normals from _load_one_scan scene_fts",
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

#     # colors
#     if color_by_normals and (n_vis is not None) and (not keep_scene_rgb):
#         pcd.colors = o3d.utility.Vector3dVector(normals_to_rgb01(n_vis))
#     elif rgb01 is not None and rgb01.shape[0] == xyz.shape[0]:
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
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cfg", default="msr3d/configs/data.yaml")
#     parser.add_argument("--split", default="train")
#     parser.add_argument("--scan_id", default="scene0000_00")

#     parser.add_argument("--voxel", type=float, default=0.0)
#     parser.add_argument("--point_size", type=float, default=2.0)
#     parser.add_argument("--no_normals", action="store_true")
#     parser.add_argument("--color_by_normals", action="store_true")
#     parser.add_argument("--keep_scene_rgb", action="store_true")
#     parser.add_argument("--normal_colors", action="store_true")
#     parser.add_argument("--orient_viewpoint", type=float, nargs=3, default=None)
#     parser.add_argument("--normals_dir", default=None)     # already in my version
#     parser.add_argument("--save_obj", action="store_true")       # ADD THIS

#     # keep this arg so your existing .sh doesn't break if it passes it


#     args = parser.parse_args()

#     cfg = OmegaConf.load(args.cfg)
#     loader = ScanNetBase(cfg, split=args.split)

#     # IMPORTANT: use loader output
#     _, one_scan = loader._load_one_scan(args.scan_id, load_inst_info=True, load_pc_info=True)

#     xyz, rgb01, normals = load_scene_from_scene_fts(one_scan)

#     use_normals_for_color = args.color_by_normals or args.normal_colors
#     draw_glyphs = (not args.no_normals) and (not args.normal_colors)

#     print(f"[info] scan_id={args.scan_id}")
#     print(f"[info] points={xyz.shape[0]} normals={normals.shape[0]}")

#     visualize_scene(
#         xyz=xyz,
#         rgb01=rgb01,
#         normals=normals if (use_normals_for_color or draw_glyphs) else None,
#         voxel=args.voxel,
#         point_size=args.point_size,
#         show_normals=draw_glyphs,
#         color_by_normals=use_normals_for_color,
#         keep_scene_rgb=args.keep_scene_rgb,
#         orient_viewpoint=None if args.orient_viewpoint is None else np.array(args.orient_viewpoint, dtype=np.float32),
#         window_name="Normals from _load_one_scan (scene_fts)",
#     )


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
import os
import argparse
import numpy as np
import open3d as o3d
from omegaconf import OmegaConf

# Adjust this import if your class/module name differs
from data.datasets.scan_data_loader import ScanDataLoader


def normalize_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals / nlen


def normals_to_rgb01(normals: np.ndarray) -> np.ndarray:
    n = normalize_normals(normals)
    rgb = 0.5 * (n + 1.0)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def orient_normals_toward_viewpoint(
    xyz: np.ndarray,
    normals: np.ndarray,
    viewpoint: np.ndarray
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = normalize_normals(normals)
    vp = np.asarray(viewpoint, dtype=np.float32).reshape(1, 3)
    v = vp - xyz
    flip = (np.sum(n * v, axis=1) < 0.0).astype(np.float32).reshape(-1, 1)
    return n * (1.0 - 2.0 * flip)


def load_scene_from_arkit(scan_data: dict):
    """
    Expects either:
      - scan_data['scene_pcds'] with shape (N, 9) = xyz + rgb[-1,1] + normals
    or:
      - scan_data['obj_pcds'] as dict[int -> (Ni, 9)] and concatenates them.

    Returns:
      xyz      : (N, 3)
      rgb01    : (N, 3)
      normals  : (N, 3)
    """
    if 'scene_pcds' in scan_data:
        scene = np.asarray(scan_data['scene_pcds'], dtype=np.float32)
    else:
        obj_pcds = scan_data.get('obj_pcds', None)
        if obj_pcds is None or len(obj_pcds) == 0:
            raise RuntimeError("scan_data has neither 'scene_pcds' nor non-empty 'obj_pcds'.")

        arrays = []
        if isinstance(obj_pcds, dict):
            iterable = obj_pcds.items()
        else:
            iterable = enumerate(obj_pcds)

        for key, arr in iterable:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 9:
                raise ValueError(f"obj_pcds[{key}] has invalid shape {arr.shape}; expected (N, 9)")
            arrays.append(arr)

        if not arrays:
            raise RuntimeError("No valid arrays found in obj_pcds.")

        scene = np.concatenate(arrays, axis=0)

    if scene.ndim != 2 or scene.shape[1] < 9:
        raise ValueError(f"scene data has invalid shape {scene.shape}; expected (N, 9)")

    xyz = scene[:, 0:3].astype(np.float32, copy=False)
    rgb_m11 = scene[:, 3:6].astype(np.float32, copy=False)  # expected in [-1, 1]
    normals = scene[:, 6:9].astype(np.float32, copy=False)

    rgb01 = np.clip((rgb_m11 + 1.0) * 0.5, 0.0, 1.0).astype(np.float32, copy=False)
    return xyz, rgb01, normals


def build_pcd_for_export(
    xyz: np.ndarray,
    rgb01: np.ndarray | None,
    normals: np.ndarray | None,
    *,
    voxel: float = 0.0,
    color_by_normals: bool = True,
    keep_scene_rgb: bool = False,
    orient_viewpoint: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
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

    if color_by_normals and (n_vis is not None) and (not keep_scene_rgb):
        pcd.colors = o3d.utility.Vector3dVector(normals_to_rgb01(n_vis))
    elif rgb01 is not None and rgb01.shape[0] == xyz.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb01, 0.0, 1.0))
    else:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel))

    return pcd


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
    window_name: str = "ARKit Scene Normals Viewer",
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

    if color_by_normals and (n_vis is not None) and (not keep_scene_rgb):
        pcd.colors = o3d.utility.Vector3dVector(normals_to_rgb01(n_vis))
    elif rgb01 is not None and rgb01.shape[0] == xyz.shape[0]:
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
    parser = argparse.ArgumentParser(description="Visualize ARKit scene normals.")
    parser.add_argument("--cfg", default="msr3d/configs/data.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--scan_id", default="41069021")

    parser.add_argument("--normals_dir", default=None, help="Accepted for bash compatibility; not used.")
    parser.add_argument("--voxel", type=float, default=0.0)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--no_normals", action="store_true", help="Do not draw normal glyphs.")
    parser.add_argument("--color_by_normals", action="store_true", help="Color points by normal direction.")
    parser.add_argument("--keep_scene_rgb", action="store_true", help="Keep RGB even when color_by_normals is enabled.")
    parser.add_argument("--normal_colors", action="store_true", help="Color by normal direction but do not draw glyphs.")
    parser.add_argument("--save_obj", action="store_true", help="Save point cloud as .ply.")
    parser.add_argument(
        "--orient_viewpoint",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional viewpoint to orient normals toward.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    loader = ScanDataLoader(cfg,dataset='3RScan')

    # Requires your loader to return scene_pcds and/or obj_pcds using the new scene_normals logic
    scan_data = loader._get_rscan_data(args.scan_id, data_type=['scene_pcds', 'obj_pcds'])

    xyz, rgb01, normals = load_scene_from_arkit(scan_data)

    use_normals_for_color = args.color_by_normals or args.normal_colors
    draw_normal_glyphs = (not args.no_normals) and (not args.normal_colors)

    if args.save_obj:
        out_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd_obj"
        os.makedirs(out_dir, exist_ok=True)

        pcd = build_pcd_for_export(
            xyz=xyz,
            rgb01=rgb01,
            normals=normals,
            voxel=args.voxel,
            color_by_normals=use_normals_for_color,
            keep_scene_rgb=args.keep_scene_rgb,
            orient_viewpoint=None if args.orient_viewpoint is None else np.array(args.orient_viewpoint, dtype=np.float32),
        )

        out_path = os.path.join(out_dir, f"{args.scan_id}.ply")
        ok = o3d.io.write_point_cloud(out_path, pcd, write_ascii=False)
        if not ok:
            raise RuntimeError(f"Failed to write PLY: {out_path}")
        print(f"[info] Saved point cloud: {out_path}")

    print(f"[info] scan_id={args.scan_id}")
    print(f"[info] points={xyz.shape[0]} normals={normals.shape[0]}")
    print(f"[info] color_by_normals={use_normals_for_color} keep_scene_rgb={args.keep_scene_rgb}")

    visualize_scene(
        xyz=xyz,
        rgb01=rgb01,
        normals=normals if (use_normals_for_color or draw_normal_glyphs) else None,
        voxel=args.voxel,
        point_size=args.point_size,
        show_normals=draw_normal_glyphs,
        color_by_normals=use_normals_for_color,
        keep_scene_rgb=args.keep_scene_rgb,
        orient_viewpoint=None if args.orient_viewpoint is None else np.array(args.orient_viewpoint, dtype=np.float32),
        window_name="Normals from ARKit scene_normals",
    )


if __name__ == "__main__":
    main()