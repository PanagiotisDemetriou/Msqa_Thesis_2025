import torch

path='/lustreFS/data/vcg/data/pdemetriou/Msqa_Thesis_2025/msr3d/MSR3D_BLIP_PNPP_ViC_LORA_TUNED/best.pth/pytorch_model.bin'
# Load the weights
model_data = torch.load(path, map_location='cpu')

# Inspect the structure
print(type(model_data))
print(model_data.keys() if isinstance(model_data, dict) else "Not a dict")

# If it's a state dict, view layer names and shapes
if isinstance(model_data, dict):
    for key, value in model_data.items():
        if hasattr(value, 'shape'):
            print(f"{key}: {value.shape}")