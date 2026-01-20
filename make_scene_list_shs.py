import os

# Folder containing the .pth files
source_folder = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment"

# Output file path
output_file = "/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/list/test.txt"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    for filename in os.listdir(source_folder):
        if filename.endswith(".pth"):
            name_without_ext = os.path.splitext(filename)[0]
            f.write(name_without_ext + "\n")
            print(f"Added {name_without_ext} to scene list.")
