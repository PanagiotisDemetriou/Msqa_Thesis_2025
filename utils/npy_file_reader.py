import numpy as np
import pprint

# 🧠 Set your .npy file path here
path = "/home/panagiotis/msqa/Msqa_Thesis_2025/SHS-Net/log/001/results_Scannet/ckpt_800/pred_normal/scene0000_00_normal.npy"

try:
    # Load the numpy array
    data = np.load(path, allow_pickle=True)
    print(f"\n✅ Successfully loaded: {path}\n")

    # If it's an ndarray, print info
    if isinstance(data, np.ndarray):
        print(f"📐 Array shape: {data.shape}")
        print(f"📊 Array dtype: {data.dtype}")
        
        # If it's small, print full contents; otherwise show a snippet
        if data.size < 500:
            print("\n🧾 Contents:")
            pprint.pprint(data)
        else:
            print("\n🔍 First few elements:")
            pprint.pprint(data.flatten()[:20])
    else:
        print("\n🧾 File contents (non-ndarray):")
        pprint.pprint(data)

except Exception as e:
    print(f"\n❌ Error loading {path}: {e}\n")
