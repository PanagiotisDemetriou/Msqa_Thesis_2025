import os
import torch

# Set your directory path here
path = r"/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/"

def get_max_points_from_pth_files(directory_path):
    max_points = 0
    file_with_max = None

    for filename in os.listdir(directory_path):
        if filename.endswith(".pth"):
            file_path = os.path.join(directory_path, filename)

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=False)

                # Ensure data[0] exists and is iterable
                if isinstance(data, (list, tuple)) and len(data) > 0:
                    points_count = len(data[0])

                    if points_count > max_points:
                        max_points = points_count
                        file_with_max = filename

                    print(f"{filename}: {points_count} points")

                else:
                    print(f"{filename}: Unexpected format (data[0] not found)")

            except Exception as e:
                print(f"{filename}: Error loading file ({e})")

    return max_points, file_with_max


if __name__ == "__main__":
    max_points, file_name = get_max_points_from_pth_files(path)

    print("\n==============================")
    print(f"Maximum points found: {max_points}")
    print(f"File with max points: {file_name}")
