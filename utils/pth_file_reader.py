import torch
import pprint
import numpy as np

# 🧠 Set your .pth file path here
path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0000_00.pth"
#path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/rscan_base/3RScan-ours-align/3RScan-ours-align/00d42bed-778d-2ac6-86a7-0e0e5f5f5660/normals.pth"
#path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd_normals/41069021.pth"

#path = "/mnt/d/Thesis/PTv3/model_best.pth"
#path = "scene_with_normals.pth"
# Load the file
try:
    data = torch.load(path, map_location="cpu",weights_only=False)  # load safely on CPU
    print(f"\n✅ Successfully loaded: {path}\n")
    
    # Pretty-print if it's a dictionary (common in model checkpoints)
    if isinstance(data, dict):
        print("📦 Keys in this .pth file:")
        pprint.pprint(list(data.keys()))
        print("\n🔍 Inspecting contents of first key (0):")
        pprint.pprint(data['meta'])
        #print(len(data['obj_normals_list']))
        # Optional: show summary of tensors under 'state_dict' if present
        if "state_dict" in data:
            print("\n🧠 Model state_dict keys:")
            pprint.pprint(list(data["state_dict"].keys()))
            
    else:
        print("\n🧾 File contents:")
        pprint.pprint(data[-1])
        print("length:", len(data[0]))
        print(data[0].shape)
        #print(np.unique(data[-1], return_counts=True))


except Exception as e:
    print(f"\n❌ Error loading {path}: {e}\n")