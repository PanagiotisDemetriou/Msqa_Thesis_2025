import torch
import numpy as np
from msr3d.modules.vision import PcdObjEncoder

def move_to_device(x, device):
    if hasattr(x, "to"):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x
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
def main():
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")

   # Load scene0000_00.pth from data
   data_path = "data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0000_00.pth"
   data = torch.load(data_path)
   data = move_to_device(data, device)
   xyz_coords, rgb_features, instance, obj_class = data
   print(f"Loaded data:")
   print(f"xyz_coords shape: {xyz_coords.shape}")
   print(xyz_coords)
   print(f"rgb_features shape: {rgb_features.shape}")
   print(rgb_features)
   print(f"instance shape: {instance.shape}")
   print(instance)
   print(f"obj_class shape: {obj_class.shape}")
   print(obj_class)
   # make the input of shape (B, num_objs, num_points, 3+3)
   obj_pcds = make_obj_pcds_from_scene(data, num_points=1024, use_rgb=False, device=device)
   print("obj_pcds:", tuple(obj_pcds.size()))
   model = PcdObjEncoder(cfg=None).to(device).eval()
   with torch.no_grad():
        obj_embeds, obj_logits = model(obj_pcds)

   print("embeddings:", tuple(obj_embeds.size()))  # (1, O, 768)
   print("logits:", tuple(obj_logits.size()))      # (1, O, 607)


if __name__ == "__main__":
    main()
