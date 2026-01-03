# import os
# import torch
# import numpy as np

# def describe_normals(nrm, idx):
#     if not isinstance(nrm, np.ndarray):
#         print(f"  obj[{idx}]: type={type(nrm)} (expected np.ndarray)")
#         return
#     print(
#         f"  obj[{idx}]: shape={nrm.shape}, "
#         f"dtype={nrm.dtype}, "
#         f"mean_norm={np.linalg.norm(nrm, axis=1).mean():.4f}"
#     )

# def main():
#     normals_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals/"

#     files = sorted(f for f in os.listdir(normals_dir) if f.endswith(".pth"))
#     print(f"Found {len(files)} normal cache files")

#     # inspect only a few files
#     for fname in files[:2]:
#         path = os.path.join(normals_dir, fname)
#         cache = torch.load(path, map_location="cpu", weights_only=False)

#         print(f"\nFile: {fname}")
#         print("Keys:", list(cache.keys()))

#         scan_id = cache.get("scan_id", "N/A")
#         obj_normals_list = cache.get("obj_normals_list", [])

#         print("scan_id:", scan_id)
#         print("num_objects:", len(obj_normals_list))

#         # show first few objects
#         for i, nrm in enumerate(obj_normals_list[:]):
#             describe_normals(nrm, i)

#         # show metadata
#         if "meta" in cache:
#             print("meta:", cache["meta"])

# if __name__ == "__main__":
#     main()
import os, torch, numpy as np

normals_dir = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals/"
fname = "scene0000_00.pth"

cache = torch.load(os.path.join(normals_dir, fname), map_location="cpu", weights_only=False)
nlist = cache["obj_normals_list"]

for i in range(min(5, len(nlist))):
    nrm = np.asarray(nlist[i])
    # magnitude of the mean normal; near 1 means normals all aligned, near 0 means mixed directions
    mean_dir_mag = np.linalg.norm(nrm.mean(axis=0))

    # rough “how many distinct directions” proxy: average absolute cosine similarity to mean direction
    m = nrm.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-12)
    cos = (nrm @ m)
    avg_abs_cos = np.mean(np.abs(cos))

    print(f"obj[{i}] pts={nrm.shape[0]} mean_dir_mag={mean_dir_mag:.4f} avg_abs_cos={avg_abs_cos:.4f}")
