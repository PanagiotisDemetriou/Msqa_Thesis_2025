import torch
import pprint

# 🧠 Set your .pth file path here
path = "/home/panagiotis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0642_01.pth"

# Load the file
try:
    data = torch.load(path, map_location="cpu",weights_only=False)  # load safely on CPU
    print(f"\n✅ Successfully loaded: {path}\n")
    
    # Pretty-print if it's a dictionary (common in model checkpoints)
    if isinstance(data, dict):
        print("📦 Keys in this .pth file:")
        pprint.pprint(list(data.keys()))
        print("\n🔍 Inspecting contents of first key (0):")
        pprint.pprint(data)
        # Optional: show summary of tensors under 'state_dict' if present
        if "state_dict" in data:
            print("\n🧠 Model state_dict keys:")
            pprint.pprint(list(data["state_dict"].keys()))
    else:
        print("\n🧾 File contents:")
        pprint.pprint(data)
        

except Exception as e:
    print(f"\n❌ Error loading {path}: {e}\n")