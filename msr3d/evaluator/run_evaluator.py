# run_msqa_eval.py
from gptscore_offline_evaluator import evaluate   # replace with path found by grep
out = evaluate("msqa_scannet_ptv3.json", phase_codename="test")
print(out)