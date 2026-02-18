# import os
# import torch
# import numpy as np
# import open3d as o3d
# from omegaconf import OmegaConf
# from data.datasets.scannet_base import ScanNetBase


# def estimate_normals_whole_scene(obj_pcds_list, k=30, orient=True):
#     """obj_pcds_list: list of (Ni,6) arrays [xyz,rgb]"""
#     if obj_pcds_list is None or len(obj_pcds_list) == 0:
#         return []

#     all_xyz = []
#     counts = []
#     for obj in obj_pcds_list:
#         xyz = obj[:, :3]
#         all_xyz.append(xyz)
#         counts.append(len(xyz))

#     scene_xyz = np.vstack(all_xyz)

#     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_xyz))
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
#     if orient:
#         pcd.orient_normals_consistent_tangent_plane(k)

#     scene_normals = np.asarray(pcd.normals).astype(np.float32)

#     obj_normals_list = []
#     start = 0
#     for c in counts:
#         end = start + c
#         obj_normals_list.append(scene_normals[start:end])
#         start = end

#     return obj_normals_list


# def main():
#     # This directory is only used to list scene ids via filenames.
#     src_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/"

#     # Normals-only cache output
#     k = 30
#     orient = True
#     save_dir = f"/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals/"
#     os.makedirs(save_dir, exist_ok=True)

#     # Build loader from config (your working approach)
#     cfg = OmegaConf.load("msr3d/configs/data.yaml")
#     loader = ScanNetBase(cfg, split="train")

#     # Collect scan_ids from filenames (sceneXXXX_YY.pth -> sceneXXXX_YY)
#     scan_ids = sorted([f[:-4] for f in os.listdir(src_dir) if f.endswith(".pth")])

#     processed, skipped, failed = 0, 0, 0

#     for scan_id in scan_ids:
#         out_path = os.path.join(save_dir, f"{scan_id}.pth")

#         # resumable
#         if os.path.exists(out_path):
#             skipped += 1
#             continue

#         try:
#             _, one_scan = loader._load_one_scan(
#                 scan_id,
#                 load_inst_info=True,
#                 load_pc_info=True
#             )

#             obj_pcds_list = one_scan.get("obj_pcds", None)
#             if obj_pcds_list is None:
#                 print(f"[WARN] {scan_id}: missing obj_pcds, skipping")
#                 failed += 1
#                 continue

#             obj_normals_list = estimate_normals_whole_scene(obj_pcds_list, k=k, orient=orient)

#             # sanity check: same object count and same per-object point count
#             if len(obj_normals_list) != len(obj_pcds_list):
#                 print(f"[WARN] {scan_id}: normals/pcds object count mismatch, skipping")
#                 failed += 1
#                 continue
#             for pcd, nrm in zip(obj_pcds_list, obj_normals_list):
#                 if len(pcd) != len(nrm):
#                     print(f"[WARN] {scan_id}: per-object point mismatch, skipping")
#                     failed += 1
#                     break
#             else:
#                 # save normals-only cache
#                 cache = {
#                     "scan_id": scan_id,
#                     "obj_normals_list": obj_normals_list,  # list of (Ni,3) float32 arrays
#                     "meta": {"method": "open3d_knn", "k": k, "orient": orient},
#                 }
#                 torch.save(cache, out_path)
#                 processed += 1

#                 if processed % 25 == 0:
#                     print(f"Processed {processed} scenes...")

#       #   except Exception as e:
#       #       print(f"[ERROR] {scan_id}: {repr(e)}")
#       #       failed += 1
#         except FileNotFoundError as e:
#          print(f"[ERROR] {scan_id}")
#          print("Missing file:", e.filename)
#          failed += 1


#     print(f"Done. processed={processed}, skipped(existing)={skipped}, failed={failed}")


   

# if __name__ == "__main__":
#     main()
import os
import torch
import numpy as np
import open3d as o3d


def estimate_normals_scene_xyz(scene_xyz: np.ndarray, k: int = 30, orient: bool = True) -> np.ndarray:
    """Same method as before: Open3D KNN normals + optional consistent tangent plane orientation."""
    if scene_xyz is None or scene_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_xyz.astype(np.float64)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(k)))
    if orient:
        pcd.orient_normals_consistent_tangent_plane(int(k))

    return np.asarray(pcd.normals).astype(np.float32)


def load_scene_tuple_xyz(pth_path: str) -> np.ndarray:
    """
    Your format: (xyz(N,3), rgb(N,3), sem(N,), inst(N,))
    """
    data = torch.load(pth_path, map_location="cpu", weights_only=False)

    if not isinstance(data, (list, tuple)) or len(data) < 1:
        raise ValueError(f"Unexpected format in {pth_path}: type={type(data)}, len={getattr(data,'__len__',None)}")

    xyz = data[0]
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()
    if not isinstance(xyz, np.ndarray):
        xyz = np.asarray(xyz)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz has unexpected shape {xyz.shape} in {pth_path}")

    return xyz.astype(np.float32)


def main():
    src_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/"
    save_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals/"
    os.makedirs(save_dir, exist_ok=True)

    k = 30
    orient = True

    scan_ids = sorted([f[:-4] for f in os.listdir(src_dir) if f.endswith(".pth")])

    processed, skipped, failed = 0, 0, 0

    for scan_id in scan_ids:
        in_path = os.path.join(src_dir, f"{scan_id}.pth")
        out_path = os.path.join(save_dir, f"{scan_id}.pth")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            scene_xyz = load_scene_tuple_xyz(in_path)
            scene_normals = estimate_normals_scene_xyz(scene_xyz, k=k, orient=orient)

            if scene_normals.shape[0] != scene_xyz.shape[0]:
                raise RuntimeError(
                    f"Normals/XYZ mismatch for {scan_id}: {scene_normals.shape} vs {scene_xyz.shape}"
                )

            cache = {
                "scan_id": scan_id,
                "scene_normals": scene_normals,  # (N,3) float32
                "meta": {"method": "open3d_knn", "k": int(k), "orient": bool(orient)},
            }
            torch.save(cache, out_path)
            processed += 1

            if processed % 25 == 0:
                print(f"Processed {processed} scenes...")

        except FileNotFoundError as e:
            print(f"[ERROR] {scan_id}: Missing file: {e.filename}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {scan_id}: {repr(e)}")
            failed += 1

    print(f"Done. processed={processed}, skipped(existing)={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
