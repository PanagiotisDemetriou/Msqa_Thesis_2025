import torch
import numpy as np
import open3d as o3d

def load_scannet_style_pth(path):
    obj = torch.load(path, map_location="cpu" , weights_only=False)
    # Your file: (xyz, rgb, label_a, label_b)
    xyz, rgb, a, b = obj
    xyz = np.asarray(xyz, dtype=np.float32)
    return xyz, rgb, a, b

def estimate_normals_open3d(xyz, k=30, orient=True):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

    # kNN-based normal estimation
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

    if orient:
        # Makes normals more consistent across the surface
        pcd.orient_normals_consistent_tangent_plane(k)

    normals = np.asarray(pcd.normals).astype(np.float32)
    return normals

# Example
pth_path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0000_00.pth"
xyz, rgb, a, b = load_scannet_style_pth(pth_path)
normals = estimate_normals_open3d(xyz, k=30, orient=True)

# Save back to a new .pth (tuple with normals appended)
torch.save((xyz, rgb, a, b, normals), "scene_with_normals.pth")
print(xyz.shape, normals.shape)  # (N,3) (N,3)
