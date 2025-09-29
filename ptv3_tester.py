import torch
import numpy as np
from torch import nn
import traceback
from pointcept.models.point_transformer_v3 import PointTransformerV3

def create_random_point_cloud(num_points=1024, include_rgb=True, grid_size=0.05):
    """Create a random point cloud for testing with Pointcept format"""
    # Random 3D coordinates 
    coords = torch.rand(num_points, 3) * 10 - 5  # Range [-5, 5]

    if include_rgb:
        # Random RGB values
        rgb = torch.rand(num_points, 3)  # Range [0, 1]
        features = torch.cat([coords, rgb], dim=1)  # 6D: xyz + rgb
    else:
        features = coords  # 3D: xyz only

    # Pointcept expects specific format
    data_dict = {
        'coord': coords,
        'feat': features,
        'grid_size': grid_size,  # Required by Pointcept
        'offset': torch.tensor([num_points]),
        'batch': torch.zeros(num_points, dtype=torch.long) # put all in the same batch for simplicity
    }

    return data_dict
def move_to_device(x, device):
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    print("Creating PointTransformerV3 base model...")

    try:
        # Create model with default parameters that work with Pointcept
        model = PointTransformerV3(
            in_channels=6,  # xyz + rgb
            # Use default parameters - don't specify cls_mode as it doesn't exist
        )

        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"Error creating model: {e}")
        

        # Try with minimal parameters
        model = PointTransformerV3()
        print(f"Model created with default parameters: {sum(p.numel() for p in model.parameters())} parameters")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)

    # Create random point cloud
    print("\nGenerating random point cloud...")
    data_dict = create_random_point_cloud(num_points=512, include_rgb=True)
    print(f"Point cloud shape: {data_dict['feat'].shape}")
    print(f"Coordinate range: [{data_dict['coord'].min():.2f}, {data_dict['coord'].max():.2f}]")
    print(f"Grid size: {data_dict['grid_size']}")


    # Create random point cloud with proper format
    print("\nGenerating random point cloud...")
    data_dict = create_random_point_cloud(num_points=512, include_rgb=True)
    print(f"Point cloud shape: {data_dict['feat'].shape}")
    print(f"Coordinate range: [{data_dict['coord'].min():.2f}, {data_dict['coord'].max():.2f}]")
    print(f"Grid size: {data_dict['grid_size']}")
    data_dict = move_to_device(data_dict, device)
    # Run forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            output_point = model(data_dict)
            print(f"✓ Forward pass successful!")
            print(f"Output feature shape: {output_point.feat.shape}")
            print(f"Output coordinate shape: {output_point.coord.shape}")
            print(f"Output feature range: [{output_point.feat.min():.3f}, {output_point.feat.max():.3f}]")

        except Exception as e:
            print(f"✗ Error during forward pass: {e}")
            import traceback
            traceback.print_exc()


def test_scannet_scene():
    data_path = "data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0000_00.pth"
    data = torch.load(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = move_to_device(data, device)
    xyz_coords, rgb_features, instance, obj_class = data
    print(f"Loaded scene:")
    features= torch.cat([torch.from_numpy(xyz_coords), torch.from_numpy(rgb_features) / 255.0], dim=1)
    custom_data = {
    'coord': torch.from_numpy(xyz_coords),
    'feat': features,
    'grid_size': 0.1,  # Important: include grid_size
    'offset': torch.tensor([len(xyz_coords)]),
    'batch': torch.zeros(len(xyz_coords), dtype=torch.long)
    }
    custom_data = move_to_device(custom_data, device)
    model = PointTransformerV3()
    model = model.to(device)
    model.eval()

    mlp=nn.Sequential(
        nn.LayerNorm(512),
        nn.Linear(512, 768),
    ).to(device)

    clf = nn.Linear(768, 607).to(device)  # Example for 40 classes

    with torch.no_grad():
        try:
            output = model(custom_data)
            obj_embeds=mlp(output.feat)
            obj_logits=clf(obj_embeds)
            print(f"✓ Scene forward pass successful!")
            print(f"Input points: {len(custom_data['coord'])}")
            print(f"Output points: {len(output.coord)}")
            print(f"Output feature shape: {output.feat.shape}")
            print(f"MLP output feature shape: {obj_embeds.shape}")
            print(f"Classifier logits shape: {obj_logits.shape}")

        except Exception as e:
            print(f"✗ Error with scene point cloud: {e}")
           
            traceback.print_exc()
    # print results 
    print(f"output: {output}")


if __name__ == "__main__":
    # Random Point Cloud
    main()
    # Scannet Scene Point Cloud
    test_scannet_scene()