import torch
import numpy as np
from torch import nn
# Import from the actual Pointcept library - base model
from pointcept.models.point_transformer_v3 import PointTransformerV3
def make_obj_pcds_from_scene(scene_tuple, num_points=1024, use_rgb=False, device="cpu"):
    coords, colors, instance_ids, sem_labels = scene_tuple  
    coords = coords.astype(np.float32)                      
    colors = colors.astype(np.float32)                      
    inst   = instance_ids.astype(np.int64)                  

    if use_rgb:
        feats = np.concatenate([coords, colors / 255.0], axis=1)  
    else:
        feats = coords                                            

    objs = []
    for inst_id in np.unique(inst):
        m = (inst == inst_id)
        pts = feats[m]                      
        if pts.shape[0] == 0:
            continue
        n = pts.shape[0]
        if n >= num_points:
            idx = np.random.choice(n, num_points, replace=False)
        else:
            idx = np.random.choice(n, num_points, replace=True)
        objs.append(pts[idx])

    if len(objs) == 0:
        raise ValueError("No objects found after grouping by instance IDs.")

    obj_pcds = torch.from_numpy(np.stack(objs, axis=0))# (O, P, D)
    obj_pcds = obj_pcds.unsqueeze(0)# (1, O, P, D)

    return obj_pcds.to(device)
def data_transform(obj_pcds):
    # take coords of each object for all object of all batches
    coords = obj_pcds[...,:3] 
    features = obj_pcds[...,3:]
    return coords,features




def test_scannet_scene():
    data_path = "../msr3d/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0000_00.pth"
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
        nn.LayerNorm(64),
        nn.Linear(64, 768),
    ).to(device)
    clf = nn.Linear(768, 607).to(device)  
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
            print(f"Instances:{len(np.unique(instance))}")
            print(f"Object Classes:{len(np.unique(obj_class))}")
        except Exception as e:
            print(f"✗ Error with scene point cloud: {e}")
            import traceback
            traceback.print_exc()
    # print results
    
    print(f"output: {output}")
    
def create_random_point_cloud(num_points=1024, include_rgb=True, grid_size=0.05):
    """Create a random point cloud for testing with Pointcept format"""
    # Random 3D coordinates in a reasonable range
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
        'batch': torch.zeros(num_points, dtype=torch.long)
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

    print("Creating PointTransformerV3 base model (Pointcept version)...")

    try:
        # Create model with default parameters that work with Pointcept
        model = PointTransformerV3(
            in_channels=6,  # xyz + rgb
            # Use default parameters - don't specify cls_mode as it doesn't exist
        )

        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Trying with minimal parameters...")

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

def test_with_different_grid_sizes():
    """Test with different grid sizes"""
    print("\n" + "="*50)
    print("Testing with different grid sizes...")

    model = PointTransformerV3()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)
    model = model.to(device)
    model.eval()

    grid_sizes = [0.01, 0.05, 0.1, 0.2]

    for grid_size in grid_sizes:
        print(f"\nTesting grid_size: {grid_size}")

        data_dict = create_random_point_cloud(num_points=256, grid_size=grid_size)
        data_dict = move_to_device(data_dict, device)
        with torch.no_grad():
            try:
                output = model(data_dict)
                print(f"✓ Success with grid_size {grid_size}")
                print(f"  Input points: {len(data_dict['coord'])}")
                print(f"  Output points: {len(output.coord)}")

            except Exception as e:
                print(f"✗ Failed with grid_size {grid_size}: {e}")

def test_custom_point_cloud():
    """Example of how to use your own point cloud data"""
    print("\n" + "="*50)
    print("Testing with custom point cloud...")

    # Example: create a simple organized point cloud
    # Create points in a 3D grid pattern
    n = 8  # Smaller grid for testing
    x = torch.linspace(-2, 2, n)
    y = torch.linspace(-2, 2, n)
    z = torch.linspace(-2, 2, n)

    # Create a grid of points
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    # Create simple RGB based on position (normalized)
    rgb = (coords + 2) / 4  # Normalize to [0, 1]
    features = torch.cat([coords, rgb], dim=1)

    custom_data = {
        'coord': coords,
        'feat': features,
        'grid_size': 0.1,  # Important: include grid_size
        'offset': torch.tensor([len(coords)]),
        'batch': torch.zeros(len(coords), dtype=torch.long)
    }
    custom_data = move_to_device(custom_data, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Custom point cloud shape: {custom_data['feat'].shape}")

    model = PointTransformerV3()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        try:
            output = model(custom_data)
            print(f"✓ Custom point cloud forward pass successful!")
            print(f"Input points: {len(custom_data['coord'])}")
            print(f"Output points: {len(output.coord)}")
            print(f"Output feature shape: {output.feat.shape}")
            print(f"Output:{output}")
        except Exception as e:
            print(f"✗ Error with custom point cloud: {e}")
            import traceback
            traceback.print_exc()

def inspect_model_parameters():
    """Inspect the actual model parameters to understand the architecture"""
    print("\n" + "="*50)
    print("Inspecting PointTransformerV3 parameters...")

    model = PointTransformerV3()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)
    model = model.to(device)
    # Check what parameters the model actually accepts
    import inspect
    sig = inspect.signature(PointTransformerV3.__init__)
    print("Available parameters:")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            default_val = param.default if param.default != inspect.Parameter.empty else "Required"
            print(f"  {param_name}: {default_val}")

if __name__ == "__main__":
    # First, inspect what parameters are available
    inspect_model_parameters()

    # Run the main test
    #main()

    # Test with different configurations
    #test_with_different_grid_sizes()
    #test_custom_point_cloud()
    test_scannet_scene()
