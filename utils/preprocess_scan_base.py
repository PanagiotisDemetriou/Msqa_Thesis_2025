import torch
import numpy as np
import os

# pth_path = "data/msqa/MSR3D_v2_pcds/scannet_base/scan_data/instance_id_to_label/scene0000_00.pth"
pth_path = "data/msqa/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/pcd_with_global_alignment/scene0000_00.pth"
data = torch.load(pth_path, map_location="cpu",weights_only=False)

print("Type:", type(data))
if isinstance(data, dict):
    print("Keys:", data.keys())
elif isinstance(data, (list, tuple)):
    print("Length:", len(data))
    if len(data) > 0:
        print("First element type:", type(data[0]))
else:
    print("Data:", data)

for k in range(3):  # just the first few
    v = data[k]
    print(f"Key {k}: type={type(v)}")
    if torch.is_tensor(v):
        print(f"  shape={tuple(v.shape)} dtype={v.dtype}")
    elif isinstance(v, dict):
        print(f"  dict keys: {list(v.keys())}")
    elif isinstance(v, np.ndarray):
        print(f"  ndarray shape={v.shape} dtype={v.dtype}")
    else:
        print("  preview:", v)

