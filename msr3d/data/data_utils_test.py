import numpy as np
import os
from pathlib import Path
import torch

def load_point_cloud(scene_id):
    """
    Load point cloud data from MSR3D dataset structure
    
    Args:
        scene_id: ID of the scene (e.g., 'scene0090_00')
        
    Returns:
        point_data: Dictionary containing:
            - xyz: (N, 3) array of point coordinates
            - rgb: (N, 3) array of point colors
            - instance_labels: (N,) array of instance IDs
            - semantic_labels: (N,) array of semantic class labels
    """
    base_path = Path('data/MSR3D_v2_pcds/scannet_base/scan_data')
    scene_path = base_path / scene_id
    
    # Load the 4 arrays
    xyz = np.load(scene_path / 'xyz.npy')  # Point coordinates
    rgb = np.load(scene_path / 'rgb.npy')  # Point colors
    instance_labels = np.load(scene_path / 'instance.npy')  # Instance IDs
    semantic_labels = np.load(scene_path / 'semantic.npy')  # Semantic labels
    
    # Combine XYZ and RGB for the model's expected format
    point_cloud = np.concatenate([xyz, rgb], axis=1)
    
    return {
        'point_cloud': point_cloud,
        'instance_labels': instance_labels,
        'semantic_labels': semantic_labels
    }

def load_point_cloud_with_objects(scene_id):
    """
    Load point cloud data and extract object information
    
    Args:
        scene_id: ID of the scene (e.g., 'scene0090_00')
        
    Returns:
        data_dict: Dictionary containing:
            - obj_fts: (N_objects, C) Object features
            - obj_masks: (N_objects,) Binary mask for valid objects
            - obj_locs: (N_objects, 6) Object locations (xyz) and orientations (xyz)
            - point_cloud: (N_points, 6) Raw point cloud with XYZ and RGB
            - instance_labels: Point-to-object mapping
            - semantic_labels: Semantic class per point
    """
    base_path = Path('data/MSR3D_v2_pcds/scannet_base/scan_data')
    scene_path = base_path / scene_id
    
    # Load the point cloud arrays
    xyz = np.load(scene_path / 'xyz.npy')  # Point coordinates
    rgb = np.load(scene_path / 'rgb.npy')  # Point colors
    instance_labels = np.load(scene_path / 'instance.npy')  # Instance IDs
    semantic_labels = np.load(scene_path / 'semantic.npy')  # Semantic labels
    
    # Combine points and colors
    point_cloud = np.concatenate([xyz, rgb], axis=1)
    
    # Extract unique objects (excluding background/ignore labels)
    unique_instances = np.unique(instance_labels)
    valid_instances = unique_instances[unique_instances != -100]
    
    obj_features = []
    obj_locations = []
    obj_masks = []
    
    for inst_id in valid_instances:
        # Get points belonging to this object
        obj_mask = instance_labels == inst_id
        obj_points = xyz[obj_mask]
        
        if len(obj_points) > 0:
            # Calculate object center and size as features
            center = obj_points.mean(axis=0)
            size = obj_points.max(axis=0) - obj_points.min(axis=0)
            
            # Basic object features (can be enhanced)
            obj_feat = np.concatenate([center, size])  # 6-dimensional feature
            
            # Object location (xyz) and dummy orientation
            obj_loc = np.concatenate([center, np.zeros(3)])  # xyz + dummy orientation
            
            obj_features.append(obj_feat)
            obj_locations.append(obj_loc)
            obj_masks.append(1)  # Mark as valid object
    
    # Convert to numpy arrays with dummy values if no objects found
    if not obj_features:
        obj_features = np.zeros((1, 6))  # Dummy object
        obj_locations = np.zeros((1, 6))
        obj_masks = np.zeros(1)
    else:
        obj_features = np.stack(obj_features)
        obj_locations = np.stack(obj_locations)
        obj_masks = np.array(obj_masks)
    
    return {
        'obj_fts': obj_features,
        'obj_masks': obj_masks,
        'obj_locs': obj_locations,
        'point_cloud': point_cloud,
        'instance_labels': instance_labels,
        'semantic_labels': semantic_labels
    }

def prepare_model_input(scene_id, question):
    """
    Prepare the complete input dictionary for the model
    
    Args:
        scene_id: Scene identifier
        question: Question text
        
    Returns:
        data_dict: Dictionary containing all required model inputs
    """
    # Load point cloud and object data
    pc_data = load_point_cloud_with_objects(scene_id)
    
    # Create the data dictionary with required fields
    data_dict = {
        'scan_id': scene_id,
        'obj_fts': torch.from_numpy(pc_data['obj_fts']).float().unsqueeze(0),  # (1, N, C)
        'obj_masks': torch.from_numpy(pc_data['obj_masks']).bool().unsqueeze(0),  # (1, N)
        'obj_locs': torch.from_numpy(pc_data['obj_locs']).float().unsqueeze(0),  # (1, N, 6)
        'img_fts': torch.zeros(1, 3, 224, 224),  # Dummy image features (should be replaced with real image if available)
        'img_masks': torch.ones(1, 1),  # Dummy image mask
        'question': question
    }
    
    return data_dict