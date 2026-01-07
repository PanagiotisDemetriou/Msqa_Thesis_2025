# TO RUN:
# python validate_arkit_normals.py
#
# This script validates that normals saved in:
#   /mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd_normals/<scene_id>.pth
# match the geometry loaded from:
#   /mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd-align/pcd-align/<scene_id>.pth
#
# It checks:
# - All points receive normals (indexing correct)
# - Normals are unit length
# - Local PCA normal agreement (abs cosine similarity)
# - Agreement with a fresh Open3D normal recompute (abs cosine similarity)
# - Optional visualization (commented)

import os
import numpy as np
import torch
import open3d as o3d


def normalize_rows(v, eps=1e-8):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, eps, None)


def load_arkit_scene_and_cache(scene_id: str,
                               pcd_dir: str,
                               normals_dir: str):
    pcd_path = os.path.join(pcd_dir, f"{scene_id}.pth")
    normals_path = os.path.join(normals_dir, f"{scene_id}.pth")

    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"Missing pcd file: {pcd_path}")
    if not os.path.exists(normals_path):
        raise FileNotFoundError(f"Missing normals file: {normals_path}")

    xyz, rgb, inst_ids = torch.load(pcd_path, map_location="cpu", weights_only=False)

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb)
    inst_ids = np.asarray(inst_ids)

    # Cast instance ids robustly (ARKIT sample shows float)
    if inst_ids.dtype.kind == "f":
        if np.max(np.abs(inst_ids - np.round(inst_ids))) > 1e-4:
            raise ValueError(f"{scene_id}: inst_ids are float but not near-integers.")
        inst_ids = np.round(inst_ids).astype(np.int64)
    else:
        inst_ids = inst_ids.astype(np.int64)

    cache = torch.load(normals_path, map_location="cpu", weights_only=False)
    normals_by_inst = cache.get("normals_by_inst", None)
    indices_by_inst = cache.get("indices_by_inst", None)
    if normals_by_inst is None or indices_by_inst is None:
        raise KeyError(f"{scene_id}: normals cache missing normals_by_inst/indices_by_inst")

    return xyz, rgb, inst_ids, normals_by_inst, indices_by_inst, cache


def reconstruct_scene_normals_from_cache(xyz, normals_by_inst, indices_by_inst):
    N = xyz.shape[0]
    scene_normals = np.full((N, 3), np.nan, dtype=np.float32)

    total_written = 0
    for inst_id, idx in indices_by_inst.items():
        idx = np.asarray(idx, dtype=np.int64)
        nrm = np.asarray(normals_by_inst[int(inst_id)], dtype=np.float32)

        if idx.ndim != 1:
            raise ValueError(f"indices for inst {inst_id} are not 1D: {idx.shape}")
        if nrm.ndim != 2 or nrm.shape[1] != 3:
            raise ValueError(f"normals for inst {inst_id} not (Ni,3): {nrm.shape}")
        if idx.shape[0] != nrm.shape[0]:
            raise ValueError(f"inst {inst_id}: idx count {idx.shape[0]} != normals count {nrm.shape[0]}")
        if np.any(idx < 0) or np.any(idx >= N):
            raise ValueError(f"inst {inst_id}: indices out of range")

        scene_normals[idx] = nrm
        total_written += idx.shape[0]

    missing = int(np.isnan(scene_normals).any(axis=1).sum())
    return scene_normals, total_written, missing


def pca_normal_for_point(xyz, nbr_idx):
    P = xyz[nbr_idx]  # (k,3)
    mu = P.mean(axis=0, keepdims=True)
    Q = P - mu
    C = (Q.T @ Q) / max(Q.shape[0], 1)
    w, V = np.linalg.eigh(C)
    n = V[:, 0]  # smallest eigenvalue
    n = n / (np.linalg.norm(n) + 1e-8)
    return n.astype(np.float32)


def validate_with_local_pca(xyz, scene_normals, sample_size=2000, knn=30, seed=0):
    N = xyz.shape[0]
    if N == 0:
        return {"count": 0, "bad": 0}

    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, N)
    sample_idx = rng.choice(N, size=sample_size, replace=False)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    cos_sims = []
    bad = 0

    for i in sample_idx:
        _, idx, _ = kdtree.search_knn_vector_3d(xyz[i], knn)
        if len(idx) < 5:
            bad += 1
            continue

        n_pca = pca_normal_for_point(xyz, np.asarray(idx, dtype=np.int64))
        n_cache = scene_normals[i]

        if not np.isfinite(n_cache).all():
            bad += 1
            continue

        cos = float(np.abs(np.dot(n_pca, n_cache) / (np.linalg.norm(n_cache) + 1e-8)))
        cos_sims.append(cos)

    if len(cos_sims) == 0:
        return {"count": 0, "bad": bad}

    cos_sims = np.asarray(cos_sims)
    return {
        "count": int(len(cos_sims)),
        "bad": int(bad),
        "mean_abs_cos": float(cos_sims.mean()),
        "median_abs_cos": float(np.median(cos_sims)),
        "p10_abs_cos": float(np.quantile(cos_sims, 0.10)),
        "p01_abs_cos": float(np.quantile(cos_sims, 0.01)),
    }


def recompute_open3d_normals(xyz, knn=30, orient=True):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    if orient:
        pcd.orient_normals_consistent_tangent_plane(knn)
    return np.asarray(pcd.normals, dtype=np.float32)


def compare_to_open3d(scene_normals_cached, scene_normals_ref):
    a = normalize_rows(scene_normals_cached.copy())
    b = normalize_rows(scene_normals_ref.copy())
    cos = np.sum(a * b, axis=1)
    cos = np.abs(cos)
    return {
        "mean_abs_cos": float(np.mean(cos)),
        "median_abs_cos": float(np.median(cos)),
        "p10_abs_cos": float(np.quantile(cos, 0.10)),
        "p01_abs_cos": float(np.quantile(cos, 0.01)),
    }


def visualize_normals(xyz, rgb, normals, stride=200, normal_len=0.05):
    idx = np.arange(0, xyz.shape[0], stride)
    pts = xyz[idx]
    cols = rgb[idx] / 255.0 if rgb.max() > 1.5 else rgb[idx]
    nrm = normals[idx]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))
    pcd.normals = o3d.utility.Vector3dVector(nrm.astype(np.float32))

    line_pts = np.vstack([pts, pts + nrm * normal_len])
    lines = np.array([[i, i + len(pts)] for i in range(len(pts))], dtype=np.int32)
    colors = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (lines.shape[0], 1))

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd, line_set])


def main():
    # Paths
    pcd_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd-align/pcd-align/"
    normals_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd_normals"

    # Pick a scene id (filename without .pth)
    # Example: if you have ".../pcd-align/pcd-align/scene_0001.pth" -> scene_id="scene_0001"
    scene_id = None

    # If you don't want to manually set scene_id, auto-pick the first file that has normals
    if scene_id is None:
        candidates = sorted([f[:-4] for f in os.listdir(pcd_dir) if f.endswith(".pth")])
        for cid in candidates:
            if os.path.exists(os.path.join(normals_dir, f"{cid}.pth")):
                scene_id = cid
                break
        if scene_id is None:
            raise RuntimeError("No scene found with both pcd and normals present.")

    xyz, rgb, inst_ids, normals_by_inst, indices_by_inst, cache = load_arkit_scene_and_cache(
        scene_id, pcd_dir, normals_dir
    )

    print(f"Scene id: {scene_id}")
    print(f"Scene points: N={xyz.shape[0]}")
    print(f"Unique instance ids: {len(np.unique(inst_ids))}")

    scene_normals, written, missing = reconstruct_scene_normals_from_cache(
        xyz, normals_by_inst, indices_by_inst
    )
    print(f"Normals reconstructed: written={written} missing={missing}")
    if missing != 0:
        raise RuntimeError("Some points did not receive normals from cache (indices mismatch).")

    norms = np.linalg.norm(scene_normals, axis=1)
    print(f"Normal magnitudes: mean={norms.mean():.4f}  min={norms.min():.4f}  max={norms.max():.4f}")
    frac_near_unit = float(np.mean(np.abs(norms - 1.0) < 0.05))
    print(f"Fraction within 0.05 of unit length: {frac_near_unit:.3f}")

    # Local PCA check
    pca_stats = validate_with_local_pca(
        xyz, scene_normals,
        sample_size=2000,
        knn=30,
        seed=0
    )
    print("\nLocal PCA consistency (abs cosine similarity):")
    for k, v in pca_stats.items():
        print(f"  {k}: {v}")

    # Independent Open3D recompute
    ref_normals = recompute_open3d_normals(xyz, knn=30, orient=True)
    ref_stats = compare_to_open3d(scene_normals, ref_normals)
    print("\nComparison to Open3D recomputed normals (abs cosine similarity):")
    for k, v in ref_stats.items():
        print(f"  {k}: {v}")

    # Optional visualization
    # visualize_normals(xyz, rgb, scene_normals, stride=200, normal_len=0.05)


if __name__ == "__main__":
    main()
