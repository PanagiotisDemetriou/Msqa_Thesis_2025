import os
import collections
import json
import pickle
import random

import jsonlines
from tqdm import tqdm
from scipy import sparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
import einops 
from pathlib import Path
import json

label_dir=json.load(open("/home/panagiotis/msqa/Msqa_Thesis_2025/data/msqa/MSR3D_v2_pcds/scannet_base/annotations/meta_data/scannetv2_raw_categories.json",'r', encoding="utf-8"))

print(f"Initial number of labels: {len(label_dir)}")

root = Path("/home/panagiotis/msqa/Msqa_Thesis_2025/data/msqa/MSR3D_v2_pcds/scannet_base/scan_data/instance_id_to_name")
for scene_file in sorted(root.glob("scene*.json")):
    scene_id = scene_file.stem 
    scene_dir=root / f"{scene_id}.json"
    labels=json.load(open(scene_dir,'r', encoding="utf-8"))
    for label in labels:
         if label not in label_dir:
               print(f"New label found: {label['label_id']} - {label['label_name']}")
               # Add the new label to the dictionary
               label_dir.append(label)

print(f"Final number of labels: {len(label_dir)}")