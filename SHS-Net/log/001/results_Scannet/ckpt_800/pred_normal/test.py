import os
import numpy as np
def _load_scene_normals_txt(scan_id: str) -> np.ndarray:
    normals_path = os.path.join("SHS-Net", "log", "001", "results_Scannet", "ckpt_800", "pred_normal", f"{scan_id}.normals")
    
    if not os.path.exists(normals_path):
        raise FileNotFoundError(f"Missing normals file: {normals_path}")

    # Text file with 3 floats per line
    normals = np.loadtxt(normals_path, dtype=np.float32)
    if normals.ndim == 1:
        normals = normals.reshape(1, 3)
    if normals.shape[1] != 3:
        raise ValueError(f"Invalid normals shape in {normals_path}: {normals.shape}")

    return normals  # (N,3)

def main():
    normals = _load_scene_normals_txt("scene0000_00")
    print("Loaded normals shape:", normals.shape)
    print("First 5 normals:\n", normals[:5])


if __name__ == "__main__":
    main()