# TO RUN:
# python validate_normals_rscan.py

import os
import numpy as np
import torch
import open3d as o3d


def load_scene(scene_dir: str):
    pcds_path = os.path.join(scene_dir, "pcds.pth")
    normals_path = os.path.join(scene_dir, "normals.pth")

    if not os.path.exists(pcds_path):
        raise FileNotFoundError(f"Missing {pcds_path}")
    if not os.path.exists(normals_path):
        raise FileNotFoundError(f"Missing {normals_path}")

    xyz, rgb, inst_ids = torch.load(pcds_path, map_location="cpu", weights_only=False)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.float32)
    inst_ids = np.asarray(inst_ids)

    cache = torch.load(normals_path, map_location="cpu", weights_only=False)

    # Expected from the script I provided:
    # cache["normals_by_inst"]: dict inst_id -> (Ni,3)
    # cache["indices_by_inst"]: dict inst_id -> indices into xyz
    normals_by_inst = cache.get("normals_by_inst", None)
    indices_by_inst = cache.get("indices_by_inst", None)

    if normals_by_inst is None or indices_by_inst is None:
        raise KeyError("normals.pth does not contain 'normals_by_inst' and 'indices_by_inst'.")

    return xyz, rgb, inst_ids, normals_by_inst, indices_by_inst, cache


def reconstruct_scene_normals_from_cache(xyz, normals_by_inst, indices_by_inst):
    """Rebuild scene_normals (N,3) in the original xyz order."""
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

    # Ensure all points got normals
    missing = np.isnan(scene_normals).any(axis=1).sum()
    return scene_normals, total_written, missing


def normalize_rows(v, eps=1e-8):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, eps, None)


def pca_normal_for_point(xyz, nbr_idx):
    """
    Compute a normal estimate using PCA on neighbors:
      normal = eigenvector of smallest eigenvalue of covariance.
    Returns a unit vector (3,).
    """
    P = xyz[nbr_idx]  # (k,3)
    mu = P.mean(axis=0, keepdims=True)
    Q = P - mu
    C = (Q.T @ Q) / max(Q.shape[0], 1)
    # eigh for symmetric
    w, V = np.linalg.eigh(C)
    n = V[:, 0]  # smallest eigenvalue
    n = n / (np.linalg.norm(n) + 1e-8)
    return n.astype(np.float32)


def validate_with_local_pca(xyz, scene_normals, sample_size=2000, knn=30, seed=0):
    """
    For a random subset of points:
      - find kNN in xyz
      - compute PCA normal
      - compare angle to cached normal via |cos|
    """
    N = xyz.shape[0]
    if N == 0:
        return None

    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, N)
    sample_idx = rng.choice(N, size=sample_size, replace=False)

    # KDTree
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    cos_sims = []
    bad = 0

    for i in sample_idx:
        # Open3D kNN query
        _, idx, _ = kdtree.search_knn_vector_3d(xyz[i], knn)
        if len(idx) < 5:
            bad += 1
            continue

        n_pca = pca_normal_for_point(xyz, np.asarray(idx, dtype=np.int64))
        n_cache = scene_normals[i]

        # Handle potential NaNs
        if not np.isfinite(n_cache).all():
            bad += 1
            continue

        # normals are sign-ambiguous; use abs cosine similarity
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
    """
    Recompute normals directly with Open3D and return (N,3).
    This is an independent reference check (not perfect, but useful).
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    if orient:
        pcd.orient_normals_consistent_tangent_plane(knn)
    return np.asarray(pcd.normals, dtype=np.float32)


def compare_to_open3d(scene_normals_cached, scene_normals_ref):
    """
    Compare cached normals to recomputed normals using abs cosine similarity.
    """
    if scene_normals_cached.shape != scene_normals_ref.shape:
        raise ValueError("Normals shape mismatch.")

    a = normalize_rows(scene_normals_cached.copy())
    b = normalize_rows(scene_normals_ref.copy())

    cos = np.sum(a * b, axis=1)
    cos = np.abs(cos)  # sign ambiguity
    return {
        "mean_abs_cos": float(np.mean(cos)),
        "median_abs_cos": float(np.median(cos)),
        "p10_abs_cos": float(np.quantile(cos, 0.10)),
        "p01_abs_cos": float(np.quantile(cos, 0.01)),
    }


def visualize_normals(xyz, rgb, normals, stride=200, normal_len=0.05):
    """
    Visualize a subsampled point cloud with normals.
    """
    idx = np.arange(0, xyz.shape[0], stride)
    pts = xyz[idx]
    cols = rgb[idx] / 255.0 if rgb.max() > 1.5 else rgb[idx]
    nrm = normals[idx]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    pcd.normals = o3d.utility.Vector3dVector(nrm)

    # Create normal line set
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
    # Change this to the folder you tested (one UUID folder)
    scene_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/rscan_base/3RScan-ours-align/3RScan-ours-align/00d42bed-778d-2ac6-86a7-0e0e5f5f5660"

    xyz, rgb, inst_ids, normals_by_inst, indices_by_inst, cache = load_scene(scene_dir)

    print(f"Scene points: N={xyz.shape[0]}")
    print(f"Unique instance ids: {len(np.unique(inst_ids))}")

    scene_normals, total_written, missing = reconstruct_scene_normals_from_cache(
        xyz, normals_by_inst, indices_by_inst
    )
    print(f"Normals reconstructed: written={total_written} missing={missing}")
    if missing != 0:
        raise RuntimeError("Some points did not receive normals from cache (indices mismatch).")

    # Check normalization quality
    norms = np.linalg.norm(scene_normals, axis=1)
    print(f"Normal magnitudes: mean={norms.mean():.4f}  min={norms.min():.4f}  max={norms.max():.4f}")
    frac_near_unit = float(np.mean(np.abs(norms - 1.0) < 0.05))
    print(f"Fraction within 0.05 of unit length: {frac_near_unit:.3f}")

    # PCA local consistency check
    pca_stats = validate_with_local_pca(
        xyz, scene_normals,
        sample_size=2000,
        knn=30,
        seed=0
    )
    print("\nLocal PCA consistency (abs cosine similarity):")
    for k, v in pca_stats.items():
        print(f"  {k}: {v}")

    # Independent Open3D recompute check (optional but recommended)
    ref_normals = recompute_open3d_normals(xyz, knn=30, orient=True)
    ref_stats = compare_to_open3d(scene_normals, ref_normals)
    print("\nComparison to Open3D recomputed normals (abs cosine similarity):")
    for k, v in ref_stats.items():
        print(f"  {k}: {v}")

    # Optional visualization
    # Uncomment to visually inspect normals (subsampled)
    # visualize_normals(xyz, rgb, scene_normals, stride=200, normal_len=0.05)


if __name__ == "__main__":
    main()
