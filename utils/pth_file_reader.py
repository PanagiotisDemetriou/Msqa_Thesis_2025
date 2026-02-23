import torch
import pprint
import numpy as np
import pandas as pd

# 🧠 Set your .pth file path here
path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0162_00.pth"
#path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/rscan_base/3RScan-ours-align/3RScan-ours-align/00d42bed-778d-2ac6-86a7-0e0e5f5f5660/pcds.pth"
#path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd-align/pcd-align/41069021.pth"

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
        pprint.pprint(len(data['obj_normals_list']))
        cnt=0
        for i in range(len(data['obj_normals_list'])):  
            cnt= cnt + data['obj_normals_list'][i].shape[0]
        print(cnt)

        #print(len(data['obj_normals_list']))
        # Optional: show summary of tensors under 'state_dict' if present
        if "state_dict" in data:
            print("\n🧠 Model state_dict keys:")
            pprint.pprint(list(data["state_dict"].keys()))
            
    else:
        print("\n🧾 File contents:")
        pprint.pprint(data)
        print("length:", len(data[0]))
        print(data[2].shape)
        print(np.unique(data[2], return_counts=True))
        print(np.unique(data[-2], return_counts=True))

        tsv_path = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/annotations/meta_data/scannetv2-labels.combined.tsv"  # adjust path
        df = pd.read_csv(tsv_path, sep="\t")

        nyu40id_to_name = (
            df[["nyu40id", "nyu40class"]]
            .drop_duplicates()
            .set_index("nyu40id")["nyu40class"]
            .to_dict()
        )

        ids, counts = np.unique(data[2], return_counts=True)

        for i, c in zip(ids, counts):
            print(f"{int(i):>2}  count={int(c):>7}  name={nyu40id_to_name.get(int(i), 'UNKNOWN')}")

        print(f"Unique instance IDs: {len(np.unique(data[-1]))}")
        print (f"Unique instance IDs: {np.unique(data[-1])}")
        print(data[0].shape)



except Exception as e:
    print(f"\n❌ Error loading {path}: {e}\n")