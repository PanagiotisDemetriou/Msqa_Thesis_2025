# convert_results_pt_to_msqa_json.py
import torch, json
from pathlib import Path
from collections import defaultdict

IN_PATH = "MSR3D_BLIP_PTPNPP_VICUNA_TUNED/eval_results/msqa_scannet/results.pt"   # change if different
OUT_PATH = "msqa_input.json"

data = torch.load(IN_PATH, map_location="cpu")
print("Loaded:", type(data), "Num items:", len(data))

# Group by dataset source field if present ('source' shown in sample)
grouped = defaultdict(list)

for i, item in enumerate(data):
    # determine dataset key: use item['source'] if present, else default to 'scannet'
    dataset = item.get("source", "scannet")
    # normalize name if needed (your MSQAEvaluator expects keys like 'scannet', 'RScan', 'ARKitScenes')
    # leave as-is for now
    inst = {
        "response_pred": item.get("response_pred") or item.get("pred") or "",
        # MSQAEvaluator expects response_gt to be a list
        "response_gt": item.get("response_gt") if isinstance(item.get("response_gt"), list) else [item.get("response_gt", "")],
        "type": item.get("type", "others"),
        "question": item.get("question", item.get("instruction", "")),
        "index": item.get("index", i)
    }
    grouped[dataset].append(inst)

# Convert defaultdict to normal dict and write JSON
out = {k: v for k, v in grouped.items()}
Path(OUT_PATH).write_text(json.dumps(out, indent=2))
print("Wrote:", OUT_PATH)
