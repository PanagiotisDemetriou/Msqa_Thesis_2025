# TO RUN
# PYTHONPATH="$PWD:$PWD/msr3d:$PWD/Pointcept_main:$PYTHONPATH" python preprocess_rscan_base.py

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
        # Can be unstable on noisy scans; set orient=False if needed.
        pcd.orient_normals_consistent_tangent_plane(k)

    return np.asarray(pcd.normals, dtype=np.float32)


def split_by_instance(scene_normals: np.ndarray, inst_ids: np.ndarray):
    """
    Build a mapping from instance id -> normals for points belonging to that instance.

    Returns:
        normals_by_inst: dict[int, np.ndarray] where each value is (Ni, 3)
        indices_by_inst: dict[int, np.ndarray] indices into the original arrays
    """
    if inst_ids.ndim != 1 or inst_ids.shape[0] != scene_normals.shape[0]:
        raise ValueError(f"inst_ids must be (N,) matching normals. Got {inst_ids.shape} vs {scene_normals.shape}")

    normals_by_inst = {}
    indices_by_inst = {}

    unique_ids = np.unique(inst_ids)
    for iid in unique_ids:
        idx = np.where(inst_ids == iid)[0]
        indices_by_inst[int(iid)] = idx
        normals_by_inst[int(iid)] = scene_normals[idx]

    return normals_by_inst, indices_by_inst


def main():
    # Root directory that contains many scene folders (UUID-like names)
    src_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/rscan_base/3RScan-ours-align/3RScan-ours-align/"

    # Parameters
    k = 30
    orient = True

    processed, skipped, failed = 0, 0, 0

    scene_folders = sorted(
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    )

    for scene_id in scene_folders:
        scene_path = os.path.join(src_dir, scene_id)
        pcds_path = os.path.join(scene_path, "pcds.pth")
        out_path = os.path.join(scene_path, "normals.pth")

        if not os.path.exists(pcds_path):
            continue

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            payload = torch.load(pcds_path, map_location="cpu", weights_only=False)

            # Expect: (xyz, rgb, inst_ids)
            if not (isinstance(payload, (tuple, list)) and len(payload) == 3):
                raise TypeError(f"{scene_id}: expected tuple/list of length 3, got {type(payload)} len={getattr(payload,'__len__',None)}")

            xyz, rgb, inst_ids = payload

            xyz = np.asarray(xyz)
            inst_ids = np.asarray(inst_ids)

            if xyz.ndim != 2 or xyz.shape[1] != 3:
                raise ValueError(f"{scene_id}: xyz must be (N,3). Got {xyz.shape}")
            if inst_ids.ndim != 1 or inst_ids.shape[0] != xyz.shape[0]:
                raise ValueError(f"{scene_id}: inst_ids must be (N,) matching xyz. Got {inst_ids.shape} vs N={xyz.shape[0]}")

            # Estimate normals for whole scene, aligned with xyz indexing
            scene_normals = estimate_normals_scene_xyz(xyz, k=k, orient=orient)

            # Split normals by instance id (object)
            normals_by_inst, indices_by_inst = split_by_instance(scene_normals, inst_ids)

            # Save
            cache = {
                "scene_id": scene_id,
                "scene_normals":scene_normals,
                # Dict: inst_id -> (Ni,3) normals (point order matches indices_by_inst)
                #"normals_by_inst": normals_by_inst,
                # Dict: inst_id -> indices into the original xyz/rgb arrays
                #"indices_by_inst": indices_by_inst,
                "meta": {
                    "method": "open3d_knn",
                    "k": k,
                    "orient": orient,
                    "source": "pcds.pth=(xyz,rgb,inst_id)",
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
