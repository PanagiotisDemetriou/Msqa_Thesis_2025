import torch
from pointcept.models.utils import batch2offset
def transform_data(obj_pcds):
   """Transform input data from Pointnet++ format to PTv3 format.
      Args:
         obj_pcds: (B, N, P, C) tensor, where B is batch size, N is number of objects,
                     P is number of points per object, and C is the number of channels (e.g., 3 for xyz).
      Returns:
         Point Object:
               coord: (B*N*P, 3) tensor of point coordinates
               feat: (B*N*P, C) tensor of point features (here C=3 for xyz)
               batch: (B*N*P,) tensor indicating batch index for each point
   """
   batch_size, num_objs, num_points, num_channels = obj_pcds.size()
   total_points = batch_size * num_objs * num_points
   point = {}
   # flattened coordinates and features
   point['coord'] = obj_pcds[..., :3].reshape(total_points, 3)  # Assuming C=3 for xyz
   point['feat'] = obj_pcds.reshape(total_points, num_channels)
   # check for normals for other datasets 
   # batch indices
   # (B, 1, 1) -> (B, 1) -> (B, 1, 1)
   batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
   # expand to (B, N, P) and then reshape to (B*N*P,)
   batch_indices = batch_indices.expand(batch_size, num_objs, num_points).reshape(total_points)
   point['batch'] = batch_indices.to(torch.long)

   # offset
   point['offset'] = batch2offset(point['batch'])

   # grid size 
   point['grid_size'] = 0.02  

   return point




def main():
   # Example usage


   B, N, P, C = 2, 4, 8, 6  # Example dimensions
   obj_pcds = torch.randn(B, N, P, C)  # Random point cloud data
   
   point_data = transform_data(obj_pcds)
   print(obj_pcds)
   print("Point Coordinates Shape:", point_data['offset'])  # Should be (B*N*P, 3)

if __name__ == "__main__":
    main()