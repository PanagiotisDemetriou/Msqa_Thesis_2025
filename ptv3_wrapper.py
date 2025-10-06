import einops
import torch
from torch import nn
import numpy as np
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3
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
class PTv3PcdObjEncoder(nn.Module):
    def __init__(self,
                 cfg,
                 embedding_size=768,
                 dropout=0.1,
                 path=None,
                 in_channels=6,# i diastasi kathe point sto input (3 gia xyz + 3 gia rgb)
                 order=("z", "z-trans"),# serialization order
                 stride=(2, 2, 2, 2),# to stride se kathe encoder stage pio mikro to stride, pio ligo to downsampling
                 enc_depths=(2, 2, 2, 6, 2),# posa transformer blocks exei kathe encoder stage
                 enc_channels=(32, 64, 128, 256, 512),# ta output channels kathe encoder stage
                 enc_num_head=(2, 4, 8, 16, 32),# posa kefales attention exei kathe encoder stage channels/16
                 enc_patch_size=(48, 48, 48, 48, 48),# to patch size se kathe encoder stage gia to serialized local attention
                 dec_depths=(2, 2, 2, 2),# posa transformer blocks exei kathe decoder stage
                 dec_channels=(64, 64, 128, 256),# ta output channels kathe decoder stage
                 dec_num_head=(4, 4, 8, 16),# posa kefales attention exei kathe decoder stage channels/16
                 dec_patch_size=(48, 48, 48, 48),# to patch size se kathe decoder stage gia to serialized local attention
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 drop_path=0.3,
                 pre_norm=True,
                 shuffle_orders=True,# an thelei na kanoume shuffle ta orders kathe fora pou kanoume pooling
                 enable_rpe=False,
                 enable_flash=True,
                 upcast_attention=False,
                 upcast_softmax=False,
                 cls_mode=False,# an einai true, mono encoder diladi classificationm, an einai false, encoder kai decoder diladi segmentation
                 freeze=True):
        super().__init__()
        self.freeze = freeze
        self.cls_mode = cls_mode
        output_channels = enc_channels[-1] if cls_mode else dec_channels[0]
        self.pcd_net = PointTransformerV3(
                 in_channels,# i diastasi kathe point sto input (3 gia xyz + 3 gia rgb)
                 order,# serialization order
                 stride,# to stride se kathe encoder stage pio mikro to stride, pio ligo to downsampling
                 enc_depths,# posa transformer blocks exei kathe encoder stage
                 enc_channels,# ta output channels kathe encoder stage
                 enc_num_head,# posa kefales attention exei kathe encoder stage channels/16
                 enc_patch_size,# to patch size se kathe encoder stage gia to serialized local attention
                 dec_depths,# posa transformer blocks exei kathe decoder stage
                 dec_channels,# ta output channels kathe decoder stage
                 dec_num_head,# posa kefales attention exei kathe decoder stage channels/16
                 dec_patch_size,# to patch size se kathe decoder stage gia to serialized local attention
                 mlp_ratio,
                 qkv_bias,
                 qk_scale,
                 attn_drop,
                 proj_drop,
                 drop_path,
                 pre_norm,
                 shuffle_orders,# an thelei na kanoume shuffle ta orders kathe fora pou kanoume pooling
                 enable_rpe,
                 enable_flash,
                 upcast_attention,
                 upcast_softmax,
                 cls_mode,# an einai true, mono encoder diladi classificationm, an einai false, encoder kai decoder diladi segmentation

        )
        self.projection_head = nn.Sequential(
               nn.LayerNorm(output_channels),
               nn.Linear(output_channels, embedding_size),
        )

        self.obj3d_clf_pre_head = self.get_mlp_head(embedding_size, 384, 607, dropout=0.3)

        self.dropout = nn.Dropout(dropout)
    def get_mlp_head(self,input_size, hidden_size, output_size, dropout=0):
        return nn.Sequential(*[
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        ])

    def data_transform(self,obj_pcds):
        coords = obj_pcds[...,:3]
        rgb_features = obj_pcds[...,3:]
        coords = coords.reshape(-1, 3)
        rgb_features = rgb_features.reshape(-1, 3)
        features= torch.cat([coords, rgb_features ], dim=1)
        print(features.size())
        batch_size, num_objs, num_points, _ = obj_pcds.size()
        points_per_obj = num_points
        total_objects = batch_size * num_objs

        # Batch indices: [0,0,...,1,1,...,2,2,...]
        batch_indices = torch.arange(total_objects, device=obj_pcds.device).repeat_interleave(points_per_obj)

        # Offsets: cumulative sum [points_per_obj, 2*points_per_obj, ...]
        offset = torch.arange(1, total_objects + 1, device=obj_pcds.device) * points_per_obj
        custom_data = {
        'coord': coords,
        'feat': features,
        'grid_size': 0.1,
        'offset': offset,
        'batch': batch_indices
        }
        return custom_data
    def output_pooling(self,obj_embeds, batch_size, num_objs, points_per_obj):
        obj_embeds = einops.rearrange(obj_embeds, 'b (o p) d -> b o p d', b=batch_size, o=num_objs, p=points_per_obj)
        obj_embeds = obj_embeds.mean(dim=2)  # Mean pooling over points

        return obj_embeds
    # def freeze_bn(self, model):
    #     for module in model.modules():
    #         if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    #             module.eval()
    def forward(self, obj_pcds, obj_locs=None, obj_masks=None, obj_sem_masks=None, **kwargs):
        batch_size, num_objs, points_per_obj, _ = obj_pcds.size()
        custom_data = self.data_transform(obj_pcds)
        #if self.freeze:
        self.freeze_bn(self.pcd_net)
        with torch.no_grad():
            obj_embeds = self.pcd_net(
                custom_data
            )
            obj_embeds = self.projection_head(obj_embeds.feat)
            obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
            obj_embeds = self.output_pooling(obj_embeds, batch_size, num_objs, points_per_obj)
            obj_embeds = obj_embeds.detach()
        #else:

        #    obj_embeds = self.pcd_net(
        #        einops.rearrange(obj_pcds, 'b o p d -> (b o) p d')
        #    )
        #    obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)

        # sem logits
        obj_embeds = self.dropout(obj_embeds)
        obj_sem_cls = self.obj3d_clf_pre_head(obj_embeds)
        return obj_embeds, obj_sem_cls

def main():
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")

   # Load scene0000_00.pth from data
   data_path = "../msr3d/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0000_00.pth"
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
   obj_pcds = make_obj_pcds_from_scene(data, num_points=1024, use_rgb=True, device=device)
   print("obj_pcds:", tuple(obj_pcds.size()))
   model = PTv3PcdObjEncoder(cfg=None).to(device).eval()
   with torch.no_grad():
        obj_embeds, obj_logits = model(obj_pcds)

   print("embeddings:", tuple(obj_embeds.size()))  # (1, O, 768)
   print(obj_embeds)
   print("logits:", tuple(obj_logits.size()))      # (1, O, 607)
   print(obj_logits)
   print(obj_pcds)

   print("===========================")

if __name__ == "__main__":
    main()