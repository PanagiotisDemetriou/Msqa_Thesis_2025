import torch
import numpy as np
from torch import nn
import traceback
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
    coords = obj_pcds[...,:3]
    rgb_features = obj_pcds[...,3:]
    coords = coords.reshape(-1, 3)          
    rgb_features = rgb_features.reshape(-1, 3)
    features= torch.cat([coords, rgb_features ], dim=1)
    print(features.size())
    custom_data = {
    'coord': coords,
    'feat': features,
    'grid_size': 0.1, 
    'offset': torch.tensor([len(coords)]),
    'batch': torch.zeros(len(coords), dtype=torch.long)
    }
    return custom_data
def move_to_device(x, device):
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x
def main():
    test_scannet_scene()

def test_scannet_scene():
    data_path = "../msr3d/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0000_00.pth"
    data = torch.load(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = move_to_device(data, device)
    xyz_coords, rgb_features, instance, obj_class = data
    print(f"Loaded scene:")

    obj_pcds = make_obj_pcds_from_scene(data, num_points=1024, use_rgb=True, device=device)
    custom_data = data_transform(obj_pcds)
    custom_data = move_to_device(custom_data, device)
    model = PointTransformerV3()
    model = model.to(device)
    model.eval()

    mlp=nn.Sequential(
        nn.LayerNorm(64),
        nn.Linear(64, 768),
    ).to(device)

    # NOTE
    # in the msr3d config it lenearly projects embeddings to 256 legnth

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

        except Exception as e:
            print(f"✗ Error with scene point cloud: {e}")

            traceback.print_exc()
    # print results
    print(f"output: {output}")
    print("With data transformation")

if __name__ == "__main__":
    main()
