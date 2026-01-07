# TO RUN:
# python build_arkit_normals_cache.py

import os
import torch
import numpy as np
import open3d as o3d


def estimate_normals_scene_xyz(scene_xyz: np.ndarray, k: int = 30, orient: bool = True) -> np.ndarray:
    """
    Estimate normals for a full scene point cloud.

    Args:
        scene_xyz: (N, 3) float array
        k: KNN for normal estimation
        orient: whether to orient normals consistently

    Returns:
        scene_normals: (N, 3) float32 array
    """
    if scene_xyz.ndim != 2 or scene_xyz.shape[1] != 3:
        raise ValueError(f"scene_xyz must be (N, 3). Got {scene_xyz.shape}")
    if scene_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_xyz))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

    if orient:
        # If you see instability on noisy scans, set orient=False.
        pcd.orient_normals_consistent_tangent_plane(k)

    return np.asarray(pcd.normals, dtype=np.float32)


def split_normals_by_instance(scene_normals: np.ndarray, inst_ids: np.ndarray):
    """
    Build mapping inst_id -> normals and inst_id -> indices into the original arrays.

    Returns:
      normals_by_inst: dict[int, (Ni,3) float32]
      indices_by_inst: dict[int, (Ni,) int64]
    """
    if inst_ids.ndim != 1 or inst_ids.shape[0] != scene_normals.shape[0]:
        raise ValueError(f"inst_ids must be (N,) matching normals. Got {inst_ids.shape} vs {scene_normals.shape}")

    normals_by_inst = {}
    indices_by_inst = {}

    unique_ids = np.unique(inst_ids)
    for iid in unique_ids:
        iid_int = int(iid)
        idx = np.where(inst_ids == iid)[0].astype(np.int64)
        indices_by_inst[iid_int] = idx
        normals_by_inst[iid_int] = scene_normals[idx]

    return normals_by_inst, indices_by_inst


def main():
    src_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd-align/pcd-align/"
    save_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd_normals"
    os.makedirs(save_dir, exist_ok=True)

    # Parameters (use the same as you validated for 3RScan unless you have a reason to change)
    k = 30
    orient = True

    pth_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".pth")])

    processed, skipped, failed = 0, 0, 0

    for fname in pth_files:
        scene_id = fname[:-4]  # drop .pth
        in_path = os.path.join(src_dir, fname)
        out_path = os.path.join(save_dir, f"{scene_id}.pth")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            payload = torch.load(in_path, map_location="cpu", weights_only=False)

            if not (isinstance(payload, (tuple, list)) and len(payload) == 3):
                raise TypeError(f"{scene_id}: expected tuple/list of length 3, got {type(payload)}")

            xyz, rgb, inst_ids = payload

            xyz = np.asarray(xyz, dtype=np.float32)
            rgb = np.asarray(rgb)  # not needed for normals; keep for metadata if desired
            inst_ids = np.asarray(inst_ids)

            # In your example inst_ids are float; convert robustly to int
            if inst_ids.dtype.kind == "f":
                # validate they are near-integer
                if np.max(np.abs(inst_ids - np.round(inst_ids))) > 1e-4:
                    raise ValueError(f"{scene_id}: inst_ids are float but not near-integers.")
                inst_ids = np.round(inst_ids).astype(np.int64)
            else:
                inst_ids = inst_ids.astype(np.int64)

            if xyz.ndim != 2 or xyz.shape[1] != 3:
                raise ValueError(f"{scene_id}: xyz must be (N,3). Got {xyz.shape}")
            if inst_ids.ndim != 1 or inst_ids.shape[0] != xyz.shape[0]:
                raise ValueError(f"{scene_id}: inst_ids must be (N,) matching xyz. Got {inst_ids.shape} vs N={xyz.shape[0]}")

            # Estimate normals for whole scene
            scene_normals = estimate_normals_scene_xyz(xyz, k=k, orient=orient)

            # Split by instance id
            normals_by_inst, indices_by_inst = split_normals_by_instance(scene_normals, inst_ids)

            # Save cache
            cache = {
                "scene_id": scene_id,
                "normals_by_inst": normals_by_inst,   # dict[int] -> (Ni,3)
                "indices_by_inst": indices_by_inst,   # dict[int] -> (Ni,)
                "meta": {
                    "method": "open3d_knn",
                    "k": k,
                    "orient": orient,
                    "source": "arkit pcd-align .pth = (xyz, rgb, inst_id)",
                    "num_points": int(xyz.shape[0]),
                    "num_instances": int(len(normals_by_inst)),
                },
            }

            torch.save(cache, out_path)
            processed += 1

            if processed % 25 == 0:
                print(f"Processed {processed} scenes...")

        except Exception as e:
            print(f"[ERROR] {scene_id}: {repr(e)}")
            failed += 1

    print(f"Done. processed={processed}, skipped(existing)={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
