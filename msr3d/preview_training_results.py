import torch, pprint

data = torch.load("MSR3D_BLIP_PTPNPP_VICUNA/eval_results/msqa_scannet/results.pt", map_location="cpu")
print(type(data))

if isinstance(data, dict):
    print("Keys:", list(data.keys())[:10])
    for k, v in data.items():
        print(k, "example entry:", v[:1] if isinstance(v, list) else v)
        break
elif isinstance(data, list):
    print("Num items:", len(data))
    pprint.pp(data[0])