# view_scannet_npy.py
import argparse, os
import numpy as np
import open3d as o3d

def load_array(path):
    arr = np.load(path)
    return arr

def normalize_colors(c):
    c = c.astype(np.float32)
    # If max > 1.5 we assume 0-255 and scale to 0-1
    if c.max() > 1.5:
        c = c / 255.0
    c = np.clip(c, 0.0, 1.0)
    return c

def main(scene_dir, downsample_voxel=None, save_ply=None, estimate_normals=False):
    xyz_path   = os.path.join(scene_dir, "coord.npy")
    color_path = os.path.join(scene_dir, "color.npy")
    normal_path= os.path.join(scene_dir, "normal.npy")

    

    xyz = load_array(xyz_path)            # shape (N,3)
    colors = load_array(color_path) if os.path.isfile(color_path) else None
    normals = load_array(normal_path) if os.path.isfile(normal_path) else None
    if scene_dir.startswith("scannet/val/") or scene_dir.startswith("scannet/train/"):
            instance_path = os.path.join(scene_dir, "instance.npy")
            seg20_path    = os.path.join(scene_dir, "segment20.npy")
            seg200_path   = os.path.join(scene_dir, "segment200.npy")
            if os.path.isfile(instance_path):
                instance = load_array(instance_path)
            if os.path.isfile(seg20_path):
                seg20 = load_array(seg20_path)
            if os.path.isfile(seg200_path):
                seg200 = load_array(seg200_path)
                
    # Ensure (N,3)
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

    # If coordinates look like millimeters, scale to meters (heuristic)
    # Typical ScanNet coordinates are already in meters; skip unless values are huge.
    if np.linalg.norm(np.nanmax(np.abs(xyz), axis=0)) > 100:  # very large -> likely mm
        print("[info] Detected large magnitudes; scaling XYZ by 1/1000 (mm -> m)")
        xyz = xyz / 1000.0

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(xyz)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    

    if estimate_normals or normals is None:
        # Estimate normals for better shading
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.normalize_normals()

    if downsample_voxel:
        print(f"[info] Voxel downsampling at {downsample_voxel} m")
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel)

    # Visualize
    o3d.visualization.draw(pcd)
    #o3d.visualization.draw_geometries([pcd])

    if save_ply:
        os.makedirs(os.path.dirname(save_ply) or ".", exist_ok=True)
        o3d.io.write_point_cloud(save_ply, pcd)
        print(f"[info] Saved: {save_ply}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", help="Folder containing coord.npy, color.npy, (optional) normal.npy or even instance.npy and seg20/200.npy")
    ap.add_argument("--down", type=float, default=None, help="Voxel size for downsampling in meters, e.g. 0.01")
    ap.add_argument("--save", type=str, default=None, help="Output PLY path to save the point cloud")
    ap.add_argument("--est", action="store_true", help="Force normal estimation")
    args = ap.parse_args()
    main(args.scene_dir, args.down, args.save, args.est)
