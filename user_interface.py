"""
MSQA Multi-Dataset Scene Viewer (Gradio + Plotly)
Drop-in replacement:
- scene loading now matches your dataloader logic for ScanNet / 3RScan / ARKit
- scene format includes:
    scene_fts   : (N,9) = [xyz, rgb(-1..1), normals]
    instance_ids
    segments
    obj_pcds
- normals visualization added:
    * color by normals
    * orient normals toward viewpoint
    * draw normals as glyphs (subsampled line segments)
- model preloads on startup
- no accordion / no show-hide details button
"""

import os
import json
import copy
import threading
from collections import defaultdict, OrderedDict

import torch
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

# ======================== Config ========================

SCANNET_ROOT_DIR = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/text_annotations/msqa/scannet"
ARKIT_ROOT_DIR   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/text_annotations/msqa/arkitscenes"
RSCAN_ROOT_DIR   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/text_annotations/msqa/rscan"

SCANNET_JSON_PATHS = {
    "train": os.path.join(SCANNET_ROOT_DIR, "msqa_scannet_train.json"),
    "val":   os.path.join(SCANNET_ROOT_DIR, "msqa_scannet_val.json"),
    "test":  os.path.join(SCANNET_ROOT_DIR, "msqa_scannet_test.json"),
}
ARKIT_JSON_PATHS = {
    "train": os.path.join(ARKIT_ROOT_DIR, "msqa_arkitscenes_train.json"),
    "val":   os.path.join(ARKIT_ROOT_DIR, "msqa_arkitscenes_val.json"),
    "test":  os.path.join(ARKIT_ROOT_DIR, "msqa_arkitscenes_test.json"),
}
RSCAN_JSON_PATHS = {
    "train": os.path.join(RSCAN_ROOT_DIR, "msqa_rscan_train.json"),
    "val":   os.path.join(RSCAN_ROOT_DIR, "msqa_rscan_val.json"),
    "test":  os.path.join(RSCAN_ROOT_DIR, "msqa_rscan_test.json"),
}

SCANNET_BASE_DIR = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/scannet_base"
ARKIT_BASE_DIR   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/ARkit_base"
RSCAN_BASE_DIR   = "/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data/data/MSR3D_v2_pcds/rscan_base"

ONLY_SHOW_SCANS_WITH_PTH = True

DATASET_SPECS = {
    "scannet": {
        "json_paths": SCANNET_JSON_PATHS,
        "base_dir": SCANNET_BASE_DIR,
    },
    "arkit": {
        "json_paths": ARKIT_JSON_PATHS,
        "base_dir": ARKIT_BASE_DIR,
    },
    "rscan": {
        "json_paths": RSCAN_JSON_PATHS,
        "base_dir": RSCAN_BASE_DIR,
    },
}

SCANNET20_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refridgerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
]
SCANNET20_TO_ID = {n: i for i, n in enumerate(SCANNET20_NAMES)}

# ======================== Model preload ========================

from msr3d.tools.interactive_service import MSR3DInteractiveService

def resolve_msr3d_experiment_path() -> str:
    candidates = [
        "msr3d/MSR3D_3DATASETS_FINAL_RESUME",
        "MSR3D_3DATASETS_FINAL_RESUME",
        "msr3d/MSR3D_BLIPT_PTv3_VIC_LORA_2",
        "MSR3D_BLIPT_PTv3_VIC_LORA_2",
    ]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "config.yaml")):
            return cand
    return "MSR3D_BLIPT_PTv3_VIC_LORA_2"


MSR3D_EXPERIMENT_PATH = resolve_msr3d_experiment_path()

MSR3D_SERVICE = None
MSR3D_LOCK = threading.Lock()
MSR3D_LOAD_LOCK = threading.Lock()
MSR3D_STATUS = {"loaded": False, "loading": False, "error": None}


def get_msr3d_service():
    global MSR3D_SERVICE
    if MSR3D_SERVICE is None:
        with MSR3D_LOAD_LOCK:
            if MSR3D_SERVICE is None:
                MSR3D_STATUS["loading"] = True
                try:
                    MSR3D_SERVICE = MSR3DInteractiveService(
                        experiment_path=MSR3D_EXPERIMENT_PATH,
                        split="test",
                    )
                    MSR3D_STATUS["loaded"] = True
                    MSR3D_STATUS["error"] = None
                except Exception as e:
                    MSR3D_STATUS["loaded"] = False
                    MSR3D_STATUS["error"] = f"{type(e).__name__}: {e}"
                    raise
                finally:
                    MSR3D_STATUS["loading"] = False
    return MSR3D_SERVICE


def warmup_model():
    try:
        get_msr3d_service()
        print("[model] preloaded successfully")
    except Exception as e:
        print(f"[model] preload failed: {type(e).__name__}: {e}")


def get_model_status_text():
    if MSR3D_STATUS["loaded"]:
        return "Model status: loaded"
    if MSR3D_STATUS["loading"]:
        return "Model status: loading..."
    if MSR3D_STATUS["error"]:
        return f"Model status: preload failed ({MSR3D_STATUS['error']})"
    return "Model status: not loaded"


# ======================== Utils ========================

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def ensure_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_nyu40_class(name: str) -> str:
    name = (name or "").strip().lower()
    if name == "refrigerator":
        return "refridgerator"
    return name


def build_nyu40_to_scannet20_map(tsv_path: str) -> np.ndarray:
    df = pd.read_csv(tsv_path, sep="\t")
    if "nyu40id" not in df.columns or "nyu40class" not in df.columns:
        raise ValueError(f"TSV missing required columns. Found: {df.columns.tolist()}")

    nyu40id_to_name = (
        df[["nyu40id", "nyu40class"]]
        .drop_duplicates()
        .dropna()
        .set_index("nyu40id")["nyu40class"]
        .to_dict()
    )

    max_id = int(max(nyu40id_to_name.keys()))
    lut = np.full((max_id + 1,), -1, dtype=np.int64)

    for nyu_id, name in nyu40id_to_name.items():
        key = _normalize_nyu40_class(name)
        if key in SCANNET20_TO_ID:
            lut[int(nyu_id)] = SCANNET20_TO_ID[key]

    return lut


def remap_nyu40_segment_to_scannet20(segment_nyu40: np.ndarray, lut: np.ndarray, ignore_index: int = -1) -> np.ndarray:
    seg = np.asarray(segment_nyu40)
    if seg.ndim != 1:
        raise ValueError(f"segment must be 1D (N,), got shape {seg.shape}")

    out = np.full(seg.shape, ignore_index, dtype=np.int64)
    valid = (seg >= 0) & (seg < lut.shape[0])
    out[valid] = lut[seg[valid].astype(np.int64)]
    out[out < 0] = ignore_index
    return out


def stable_seed_from_key(*parts):
    return abs(hash("||".join(map(str, parts)))) % (2**32)


def rgb_m11_to_rgb01(rgb_m11: np.ndarray) -> np.ndarray:
    rgb_m11 = np.asarray(rgb_m11, dtype=np.float32).reshape(-1, 3)
    return np.clip((rgb_m11 + 1.0) * 0.5, 0.0, 1.0)


def rgb01_to_plotly_strings(rgb01: np.ndarray):
    rgb = np.clip(np.asarray(rgb01, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def hash_colors_for_labels(labels, seed=123):
    labels = np.asarray(labels).reshape(-1)
    rng = np.random.default_rng(seed)
    uniq = np.unique(labels)
    table = {}
    for u in uniq:
        col = rng.random(3) * 0.8 + 0.2
        table[int(u)] = col.astype(np.float32)
    return np.array([table[int(x)] for x in labels], dtype=np.float32)


def normalize_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals / nlen


def normals_to_rgb01(normals: np.ndarray) -> np.ndarray:
    n = normalize_normals(normals)
    rgb = 0.5 * (n + 1.0)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def orient_normals_toward_viewpoint(
    xyz: np.ndarray,
    normals: np.ndarray,
    viewpoint: np.ndarray
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = normalize_normals(normals)
    vp = np.asarray(viewpoint, dtype=np.float32).reshape(1, 3)
    v = vp - xyz
    flip = (np.sum(n * v, axis=1) < 0.0).astype(np.float32).reshape(-1, 1)
    return n * (1.0 - 2.0 * flip)


# ======================== Orientation / overlays ========================

def get_view_vector_from_orientation(orientation):
    o = ensure_np(orientation).astype(np.float32).reshape(-1)

    if o.size == 4:
        q = o.copy()
        try:
            yaw = R.from_quat([q[0], q[1], q[2], q[3]]).as_euler("xyz", degrees=False)[-1]
        except Exception:
            yaw = 0.0

        if abs(q[3]) < 0.2 and abs(q[0]) > 0.2:
            try:
                yaw2 = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=False)[-1]
                yaw = yaw2
            except Exception:
                pass

        d = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        return d / (np.linalg.norm(d) + 1e-12)

    if o.size == 3:
        n = float(np.linalg.norm(o))
        if 0.5 <= n <= 1.5:
            return (o / (n + 1e-12)).astype(np.float32)
        yaw = float(o[-1])
        d = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
        return d / (np.linalg.norm(d) + 1e-12)

    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _normed(v):
    v = np.asarray(v, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(v)) + 1e-12
    return v / n


def aabb_edges(minb, maxb):
    minb = np.asarray(minb, dtype=np.float32)
    maxb = np.asarray(maxb, dtype=np.float32)

    x0, y0, z0 = minb
    x1, y1, z1 = maxb

    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float32)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [(corners[i], corners[j]) for i, j in edges]


def build_instance_bboxes_as_traces(xyz, instance_labels, max_boxes=200, seed=123):
    if instance_labels is None:
        return []

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    inst = np.asarray(instance_labels).reshape(-1)
    uniq = np.unique(inst)

    uniq_cols = hash_colors_for_labels(uniq, seed=seed)
    col_tab = {int(u): uniq_cols[i] for i, u in enumerate(uniq)}

    traces = []
    if len(uniq) > max_boxes:
        uniq = uniq[:max_boxes]

    for u in uniq:
        mask = (inst == u)
        if not np.any(mask):
            continue
        pts = xyz[mask]
        if pts.shape[0] < 2:
            continue

        minb = pts.min(0)
        maxb = pts.max(0)
        segs = aabb_edges(minb, maxb)
        col = col_tab[int(u)]

        xs, ys, zs = [], [], []
        for p0, p1 in segs:
            xs += [p0[0], p1[0], None]
            ys += [p0[1], p1[1], None]
            zs += [p0[2], p1[2], None]

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=4, color=f"rgb({int(col[0]*255)},{int(col[1]*255)},{int(col[2]*255)})"),
            showlegend=False,
        ))

    return traces


def build_instance_to_label_map(instance_ids, semantic_segments=None):
    inst = np.asarray(instance_ids).reshape(-1)
    if semantic_segments is None:
        return {int(u): f"instance {int(u)}" for u in np.unique(inst)}

    seg = np.asarray(semantic_segments).reshape(-1)
    labels = {}
    for u in np.unique(inst):
        mask = inst == u
        vals = seg[mask]
        vals = vals[vals >= 0]
        if vals.size == 0:
            labels[int(u)] = f"instance {int(u)}"
            continue
        counts = np.bincount(vals.astype(np.int64))
        cls_id = int(np.argmax(counts))
        if 0 <= cls_id < len(SCANNET20_NAMES):
            labels[int(u)] = SCANNET20_NAMES[cls_id]
        else:
            labels[int(u)] = f"instance {int(u)}"
    return labels


def clean_instance_to_label_map(inst_to_label):
    labels = {}
    for k, v in (inst_to_label or {}).items():
        try:
            inst_id = int(k)
        except Exception:
            continue
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().tolist()
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        labels[inst_id] = str(v)
    return labels


def normalize_search_text(text: str) -> str:
    return " ".join((text or "").lower().replace("_", " ").replace("-", " ").split())


QUESTION_OBJECT_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "me", "i", "you",
    "your", "there", "is", "are", "am", "be", "being", "been", "was", "were",
    "do", "does", "did", "can", "could", "should", "would", "will", "what",
    "which", "where", "when", "who", "why", "how", "many", "much", "of", "to",
    "in", "on", "at", "by", "for", "from", "with", "without", "and", "or",
    "if", "than", "then", "near", "nearer", "nearest", "left", "right",
    "front", "back", "behind", "around", "direction", "clock", "oclock",
    "o", "go", "turn", "find", "see", "use", "used", "want", "need", "based",
    "provide", "please", "relationship", "between", "comparison", "located",
}


def simple_word_tokens(text: str):
    cleaned = []
    for ch in normalize_search_text(text):
        cleaned.append(ch if ch.isalnum() else " ")
    toks = []
    for tok in "".join(cleaned).split():
        if tok in QUESTION_OBJECT_STOPWORDS or len(tok) < 3:
            continue
        if tok.endswith("ies") and len(tok) > 4:
            tok = tok[:-3] + "y"
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
            tok = tok[:-1]
        toks.append(tok)
    return set(toks)


def instance_ids_matching_question(payload, question_text: str, max_matches: int = 8):
    q_tokens = simple_word_tokens(question_text)
    if not q_tokens:
        return []

    matches = []
    inst_to_label = payload.get("instance_to_label", {}) or {}
    for inst_id, label in inst_to_label.items():
        label_tokens = simple_word_tokens(label)
        if label_tokens and q_tokens.intersection(label_tokens):
            matches.append(int(inst_id))

    return list(dict.fromkeys(matches))[: int(max_matches)]


def build_target_bbox_traces(payload, question_text: str, max_matches: int = 8):
    target_ids = instance_ids_matching_question(payload, question_text, max_matches=max_matches)
    if not target_ids:
        return []

    xyz = np.asarray(payload["xyz"], dtype=np.float32).reshape(-1, 3)
    inst = np.asarray(payload["instance_ids"]).reshape(-1)
    inst_to_label = payload.get("instance_to_label", {}) or {}

    traces = []
    for inst_id in target_ids:
        pts = xyz[inst == inst_id]
        if pts.shape[0] < 2:
            continue

        minb = pts.min(0)
        maxb = pts.max(0)
        xs, ys, zs = [], [], []
        for p0, p1 in aabb_edges(minb, maxb):
            xs += [p0[0], p1[0], None]
            ys += [p0[1], p1[1], None]
            zs += [p0[2], p1[2], None]

        label = inst_to_label.get(int(inst_id), f"instance {inst_id}")
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=9, color="rgb(255,230,0)"),
            name=f"target: {label} ({inst_id})",
            showlegend=True,
        ))

    return traces


def make_world_axis_traces(origin, axis_len=1.0):
    origin = np.asarray(origin, dtype=np.float32).reshape(3)
    O = origin
    X = origin + np.array([axis_len, 0, 0], dtype=np.float32)
    Y = origin + np.array([0, axis_len, 0], dtype=np.float32)
    Z = origin + np.array([0, 0, axis_len], dtype=np.float32)

    def line(p0, p1, color):
        return go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode="lines",
            line=dict(width=8, color=color),
            showlegend=False,
        )

    return [line(O, X, "red"), line(O, Y, "green"), line(O, Z, "blue")]


def make_situation_arrow_trace(
    location, orientation,
    scale=0.8,
    zoff=0.15,
    head_len_ratio=0.50,
    head_radius_ratio=0.50,
    tip_push_ratio=0.25,
    flatten_xy=True,
):
    loc = np.asarray(location, dtype=np.float32).reshape(3)

    d = _normed(get_view_vector_from_orientation(orientation))
    if flatten_xy:
        d = _normed(np.array([d[0], d[1], 0.0], dtype=np.float32))

    base = loc.copy()
    base[2] += float(zoff)
    tip = base + d * float(scale)

    shaft = go.Scatter3d(
        x=[base[0], tip[0]],
        y=[base[1], tip[1]],
        z=[base[2], tip[2]],
        mode="lines",
        line=dict(width=10, color="rgb(255,165,0)"),
        showlegend=False,
    )

    head_len = float(scale) * float(head_len_ratio)
    tip_push = float(tip_push_ratio) * head_len
    cone_tip = tip + d * tip_push
    sizeref = float(head_radius_ratio) * float(scale)

    head = go.Cone(
        x=[cone_tip[0]], y=[cone_tip[1]], z=[cone_tip[2]],
        u=[d[0] * head_len], v=[d[1] * head_len], w=[d[2] * head_len],
        anchor="tip",
        sizemode="absolute",
        sizeref=sizeref,
        showscale=False,
        colorscale=[[0, "rgb(255,165,0)"], [1, "rgb(255,165,0)"]],
        opacity=0.95,
    )

    return [shaft, head]


def build_normal_glyph_traces(
    xyz: np.ndarray,
    normals: np.ndarray,
    *,
    glyph_scale: float = 0.15,
    max_glyphs: int = 4000,
    orient_viewpoint=None,
):
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    normals = normalize_normals(normals)

    if orient_viewpoint is not None:
        normals = orient_normals_toward_viewpoint(xyz, normals, np.asarray(orient_viewpoint, dtype=np.float32))

    N = xyz.shape[0]
    if N > max_glyphs:
        sel = np.random.default_rng(0).choice(N, size=max_glyphs, replace=False)
        xyz = xyz[sel]
        normals = normals[sel]

    tips = xyz + normals * float(glyph_scale)

    xs, ys, zs = [], [], []
    for p0, p1 in zip(xyz, tips):
        xs += [p0[0], p1[0], None]
        ys += [p0[1], p1[1], None]
        zs += [p0[2], p1[2], None]

    return [
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=2, color="rgb(255,255,0)"),
            showlegend=False,
        )
    ]


# ======================== Dataset path helpers ========================

SCANNET_NYU40_LUT = build_nyu40_to_scannet20_map(
    os.path.join(SCANNET_BASE_DIR, "annotations/meta_data/scannetv2-labels.combined.tsv")
)


def resolve_primary_scene_path(dataset_name: str, scan_id: str) -> str:
    if dataset_name == "scannet":
        return os.path.join(SCANNET_BASE_DIR, "scan_data", "pcd_with_global_alignment", f"{scan_id}.pth")
    if dataset_name == "arkit":
        return os.path.join(ARKIT_BASE_DIR, "scan_data", "pcd-align", f"{scan_id}.pth")
    if dataset_name == "rscan":
        return os.path.join(RSCAN_BASE_DIR, "3RScan-ours-align", scan_id, "pcds.pth")
    raise ValueError(dataset_name)


def scene_exists(dataset_name: str, scan_id: str) -> bool:
    return os.path.exists(resolve_primary_scene_path(dataset_name, scan_id))


# ======================== Load + merge splits ========================

DATA_BY_DATASET = {}
SCAN_TO_INDICES_BY_DATASET = {}
AVAILABLE_SCANS_BY_DATASET = {}


def build_scan_index(data):
    scan_to_indices = defaultdict(list)
    for i, qa in enumerate(data):
        scan_to_indices[qa["scan_id"]].append(i)
    return dict(scan_to_indices)


for dname, spec in DATASET_SPECS.items():
    data = []
    for split_name, path in spec["json_paths"].items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing JSON for dataset '{dname}' split '{split_name}': {path}")
        items = load_json(path)
        for it in items:
            it = dict(it)
            it["split"] = split_name
            data.append(it)

    DATA_BY_DATASET[dname] = data
    SCAN_TO_INDICES_BY_DATASET[dname] = build_scan_index(data)

    all_scans = sorted(SCAN_TO_INDICES_BY_DATASET[dname].keys())
    if ONLY_SHOW_SCANS_WITH_PTH:
        avail = [sid for sid in all_scans if scene_exists(dname, sid)]
    else:
        avail = all_scans

    if len(avail) == 0:
        raise RuntimeError(f"No scan_ids available for dataset '{dname}'. Check JSON paths and scene roots.")
    AVAILABLE_SCANS_BY_DATASET[dname] = avail


# ======================== QA helpers ========================

def qa_label(dataset_name: str, i: int) -> str:
    qa = DATA_BY_DATASET[dataset_name][i]
    sit = (qa.get("situation", "") or "").strip().replace("\n", " ")
    if len(sit) > 90:
        sit = sit[:87] + "..."
    return f"{i} | {qa['scan_id']} | {qa.get('split','?')} | {sit}"


def qa_choices_for_scan(dataset_name: str, scan_id: str, split_filter: str):
    inds = SCAN_TO_INDICES_BY_DATASET[dataset_name].get(scan_id, [])
    if split_filter != "all":
        inds = [i for i in inds if DATA_BY_DATASET[dataset_name][i].get("split") == split_filter]
    return [(qa_label(dataset_name, i), i) for i in inds]


def question_label(dataset_name: str, i: int) -> str:
    qa = DATA_BY_DATASET[dataset_name][i]
    question = (qa.get("question", "") or "").strip().replace("\n", " ")
    if len(question) > 110:
        question = question[:107] + "..."
    return f"{i} | {question}"


def question_choices_for_scan(dataset_name: str, scan_id: str, split_filter: str):
    inds = SCAN_TO_INDICES_BY_DATASET[dataset_name].get(scan_id, [])
    if split_filter != "all":
        inds = [i for i in inds if DATA_BY_DATASET[dataset_name][i].get("split") == split_filter]
    return [(question_label(dataset_name, i), i) for i in inds]


def question_text_for_idx(dataset_name: str, global_idx) -> str:
    if global_idx is None:
        return ""
    qa = DATA_BY_DATASET[dataset_name][int(global_idx)]
    return qa.get("question", "") or ""


# ======================== Scene loaders matching your dataloader ========================

def load_scannet_scene(scan_id: str):
    pcd_path = os.path.join(SCANNET_BASE_DIR, "scan_data", "pcd_with_global_alignment", f"{scan_id}.pth")
    normals_path = os.path.join(SCANNET_BASE_DIR, "scan_data", "pcd_normals", f"{scan_id}.pth")

    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"Missing PTH: {pcd_path}")
    if not os.path.exists(normals_path):
        raise FileNotFoundError(f"Missing normals PTH: {normals_path}")

    pcd_data = torch.load(pcd_path, weights_only=False)
    normals_dict = torch.load(normals_path, weights_only=False)

    points = ensure_np(pcd_data[0]).astype(np.float32)
    colors = ensure_np(pcd_data[1]).astype(np.float32)
    instance_labels = ensure_np(pcd_data[-1]).astype(np.int32)
    segments_nyu40 = ensure_np(pcd_data[2]).astype(np.int64)
    scene_normals = ensure_np(normals_dict["scene_normals"]).astype(np.float32)

    colors_m11 = colors / 127.5 - 1.0
    pcds = np.concatenate([points, colors_m11], axis=1)
    scene_fts = np.concatenate([pcds, scene_normals], axis=1)

    segments_scannet20 = remap_nyu40_segment_to_scannet20(
        segments_nyu40, SCANNET_NYU40_LUT, ignore_index=-1
    )

    obj_pcds = {}
    uniq_inst = np.unique(instance_labels)
    for inst_id in uniq_inst:
        mask = (instance_labels == inst_id)
        if np.any(mask):
            obj_pcds[int(inst_id)] = scene_fts[mask]

    instance_to_label = build_instance_to_label_map(instance_labels, segments_scannet20)

    return {
        "scene_fts": scene_fts,            # (N,9) [xyz rgb(-1,1) normals]
        "instance_ids": instance_labels,   # matches your loader
        "segments": segments_scannet20,    # matches your loader
        "obj_pcds": obj_pcds,              # object clouds in same feature space
        "instance_to_label": instance_to_label,
    }


def load_rscan_scene(scan_id: str):
    pcd_path = os.path.join(RSCAN_BASE_DIR, "3RScan-ours-align", scan_id, "pcds.pth")
    normals_path = os.path.join(RSCAN_BASE_DIR, "3RScan-ours-align", scan_id, "normals.pth")
    inst_to_label_path = os.path.join(RSCAN_BASE_DIR, "3RScan-ours-align", scan_id, "inst_to_label.pth")

    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"Missing PTH: {pcd_path}")
    if not os.path.exists(normals_path):
        raise FileNotFoundError(f"Missing normals PTH: {normals_path}")
    if not os.path.exists(inst_to_label_path):
        raise FileNotFoundError(f"Missing inst_to_label PTH: {inst_to_label_path}")

    pcd_data = torch.load(pcd_path, weights_only=False)
    normals_dict = torch.load(normals_path, weights_only=False)
    inst_to_label = torch.load(inst_to_label_path, weights_only=False)

    points = ensure_np(pcd_data[0]).astype(np.float32)
    colors = ensure_np(pcd_data[1]).astype(np.float32)
    instance_labels = ensure_np(pcd_data[2]).astype(np.int32)
    scene_normals = ensure_np(normals_dict["scene_normals"]).astype(np.float32)

    colors_m11 = colors / 127.5 - 1.0
    pcds = np.concatenate([points, colors_m11], axis=1)
    scene_fts = np.concatenate([pcds, scene_normals], axis=1)

    # matches your current loader behavior
    segments = instance_labels.copy()

    obj_pcds = {}
    for inst_id in inst_to_label.keys():
        if not isinstance(inst_id, int):
            continue
        mask = (instance_labels == inst_id)
        if np.any(mask):
            obj_pcds[int(inst_id)] = scene_fts[mask]

    return {
        "scene_fts": scene_fts,
        "instance_ids": instance_labels,
        "segments": segments,
        "obj_pcds": obj_pcds,
        "instance_to_label": clean_instance_to_label_map(inst_to_label),
    }


def load_arkit_scene(scan_id: str):
    pcd_path = os.path.join(ARKIT_BASE_DIR, "scan_data", "pcd-align", f"{scan_id}.pth")
    normals_path = os.path.join(ARKIT_BASE_DIR, "scan_data", "pcd_normals", f"{scan_id}.pth")
    inst_to_label_path = os.path.join(ARKIT_BASE_DIR, "scan_data", "instance_id_to_label", f"{scan_id}_inst_to_label.pth")

    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"Missing PTH: {pcd_path}")
    if not os.path.exists(normals_path):
        raise FileNotFoundError(f"Missing normals PTH: {normals_path}")
    if not os.path.exists(inst_to_label_path):
        raise FileNotFoundError(f"Missing inst_to_label PTH: {inst_to_label_path}")

    pcd_data = torch.load(pcd_path, weights_only=False)
    normals_dict = torch.load(normals_path, weights_only=False)
    inst_to_label = torch.load(inst_to_label_path, weights_only=False)

    points = ensure_np(pcd_data[0]).astype(np.float32)
    colors = ensure_np(pcd_data[1]).astype(np.float32)
    instance_labels = ensure_np(pcd_data[2]).astype(np.int32)
    scene_normals = ensure_np(normals_dict["scene_normals"]).astype(np.float32)

    colors_m11 = colors / 127.5 - 1.0
    pcds = np.concatenate([points, colors_m11], axis=1)
    scene_fts = np.concatenate([pcds, scene_normals], axis=1)

    # matches your current loader behavior
    segments = instance_labels.copy()

    obj_pcds = {}
    for inst_id in inst_to_label.keys():
        if not isinstance(inst_id, int):
            continue
        mask = (instance_labels == inst_id)
        if mask.sum() < 10:
            continue
        obj_pcds[int(inst_id)] = scene_fts[mask]

    return {
        "scene_fts": scene_fts,
        "instance_ids": instance_labels,
        "segments": segments,
        "obj_pcds": obj_pcds,
        "instance_to_label": clean_instance_to_label_map(inst_to_label),
    }


def load_scene_like_dataloader(dataset_name: str, scan_id: str):
    if dataset_name == "scannet":
        return load_scannet_scene(scan_id)
    if dataset_name == "rscan":
        return load_rscan_scene(scan_id)
    if dataset_name == "arkit":
        return load_arkit_scene(scan_id)
    raise ValueError(dataset_name)


def scene_components_from_scene_fts(one_scan: dict):
    if "scene_fts" not in one_scan:
        raise RuntimeError("one_scan missing key 'scene_fts'")

    scene = np.asarray(one_scan["scene_fts"], dtype=np.float32)
    if scene.ndim != 2 or scene.shape[1] < 9:
        raise ValueError(f"scene_fts invalid shape {scene.shape}; expected (N,9)")

    xyz = scene[:, 0:3].astype(np.float32, copy=False)
    rgb_m11 = scene[:, 3:6].astype(np.float32, copy=False)
    normals = scene[:, 6:9].astype(np.float32, copy=False)
    rgb01 = rgb_m11_to_rgb01(rgb_m11)

    return xyz, rgb01, rgb_m11, normals


# ======================== Scene cache ========================

RAW_SCENE_CACHE = OrderedDict()
DOWNSAMPLED_CACHE = OrderedDict()
BBOX_CACHE = OrderedDict()

MAX_RAW_SCENES = 16
MAX_DOWNSAMPLED = 48
MAX_BBOX_CACHE = 96


def cache_set_lru(cache: OrderedDict, key, value, max_size: int):
    if key in cache:
        cache.pop(key)
    cache[key] = value
    while len(cache) > max_size:
        cache.popitem(last=False)


def cache_get_lru(cache: OrderedDict, key):
    if key not in cache:
        return None
    val = cache.pop(key)
    cache[key] = val
    return val


def get_raw_scene(dataset_name: str, idx: int):
    qa = DATA_BY_DATASET[dataset_name][idx]
    scan_id = qa["scan_id"]
    key = (dataset_name, scan_id)

    cached = cache_get_lru(RAW_SCENE_CACHE, key)
    if cached is not None:
        return cached

    one_scan = load_scene_like_dataloader(dataset_name, scan_id)

    xyz, rgb01, rgb_m11, normals = scene_components_from_scene_fts(one_scan)

    payload = {
        "scan_id": scan_id,
        "scene_fts": np.asarray(one_scan["scene_fts"], dtype=np.float32),
        "instance_ids": np.asarray(one_scan["instance_ids"]).reshape(-1).astype(np.int32),
        "segments": np.asarray(one_scan["segments"]).reshape(-1).astype(np.int64),
        "obj_pcds": one_scan["obj_pcds"],
        "instance_to_label": one_scan.get("instance_to_label", {}),
        "xyz": xyz,
        "rgb01": rgb01,
        "rgb_m11": rgb_m11,
        "normals": normals,
    }
    cache_set_lru(RAW_SCENE_CACHE, key, payload, MAX_RAW_SCENES)
    return payload


def get_downsampled_scene(dataset_name: str, idx: int, max_points: int):
    qa = DATA_BY_DATASET[dataset_name][idx]
    scan_id = qa["scan_id"]
    cache_key = (dataset_name, scan_id, int(max_points))

    cached = cache_get_lru(DOWNSAMPLED_CACHE, cache_key)
    if cached is not None:
        return cached, cache_key

    raw = get_raw_scene(dataset_name, idx)

    xyz = raw["xyz"]
    rgb01 = raw["rgb01"]
    rgb_m11 = raw["rgb_m11"]
    normals = raw["normals"]
    instance_ids = raw["instance_ids"]
    segments = raw["segments"]

    N = xyz.shape[0]
    max_points = int(max_points)

    if max_points > 0 and N > max_points:
        rng = np.random.default_rng(stable_seed_from_key(dataset_name, scan_id, max_points))
        sel = rng.choice(N, size=max_points, replace=False)
        xyz = xyz[sel]
        rgb01 = rgb01[sel]
        rgb_m11 = rgb_m11[sel]
        normals = normals[sel]
        instance_ids = instance_ids[sel]
        segments = segments[sel]
        scene_fts = np.concatenate([xyz, rgb_m11, normals], axis=1)
    else:
        scene_fts = raw["scene_fts"]

    rgb_colors = rgb01_to_plotly_strings(rgb01)
    instance_colors = rgb01_to_plotly_strings(hash_colors_for_labels(instance_ids, seed=123))
    segment_colors = rgb01_to_plotly_strings(hash_colors_for_labels(segments, seed=456))

    normals_unit = normalize_normals(normals)
    normal_rgb01 = normals_to_rgb01(normals_unit)
    normal_colors = rgb01_to_plotly_strings(normal_rgb01)

    qa_item = DATA_BY_DATASET[dataset_name][idx]
    location = qa_item.get("location", None)
    orientation = qa_item.get("orientation", None)

    if isinstance(location, dict):
        location = [location.get("x", 0.0), location.get("y", 0.0), location.get("z", 0.0)]

    if isinstance(orientation, dict):
        if all(k in orientation for k in ["_x", "_y", "_z", "_w"]):
            orientation = [orientation["_x"], orientation["_y"], orientation["_z"], orientation["_w"]]
        elif all(k in orientation for k in ["x", "y", "z"]):
            orientation = [orientation["x"], orientation["y"], orientation["z"]]

    payload = {
        "scan_id": scan_id,
        "split": qa_item.get("split", "unknown"),
        "situation": qa_item.get("situation", ""),
        "scene_fts": scene_fts,
        "xyz": xyz.astype(np.float32),
        "rgb01": rgb01.astype(np.float32),
        "rgb_m11": rgb_m11.astype(np.float32),
        "normals": normals.astype(np.float32),
        "instance_ids": instance_ids.astype(np.int32),
        "segments": segments.astype(np.int64),
        "obj_pcds": raw["obj_pcds"],
        "instance_to_label": raw.get("instance_to_label", {}),
        "rgb_colors": rgb_colors,
        "instance_colors": instance_colors,
        "segment_colors": segment_colors,
        "normal_colors": normal_colors,
        "center": xyz.mean(axis=0).astype(np.float32),
        "location": location,
        "orientation": orientation,
    }

    cache_set_lru(DOWNSAMPLED_CACHE, cache_key, payload, MAX_DOWNSAMPLED)
    return payload, cache_key


def get_bbox_traces(dataset_name: str, idx: int, max_points: int, max_boxes: int):
    qa = DATA_BY_DATASET[dataset_name][idx]
    scan_id = qa["scan_id"]
    key = (dataset_name, scan_id, int(max_points), int(max_boxes))

    cached = cache_get_lru(BBOX_CACHE, key)
    if cached is not None:
        return cached

    payload, _ = get_downsampled_scene(dataset_name, idx, max_points)
    traces = build_instance_bboxes_as_traces(payload["xyz"], payload["instance_ids"], max_boxes=int(max_boxes), seed=123)

    cache_set_lru(BBOX_CACHE, key, traces, MAX_BBOX_CACHE)
    return traces


# ======================== Figure helpers ========================

def make_base_figure(payload, point_size: float):
    xyz = payload["xyz"]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        marker=dict(
            size=float(point_size),
            color=payload["rgb_colors"],
            opacity=1.0,
        ),
        showlegend=False,
    ))

    title = f"{payload['scan_id']} | split: {payload.get('split','?')}"
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="black",
        ),
        paper_bgcolor="black",
        font=dict(color="white"),
        uirevision=f"{payload['scan_id']}::{payload.get('split','?')}",
    )
    return fig


def apply_style_and_overlays(
    fig,
    payload,
    dataset_name,
    idx,
    max_points,
    *,
    color_mode: str,
    point_size: float,
    show_boxes: bool,
    show_target_box: bool,
    target_box_max_matches: int,
    show_axis: bool,
    show_arrow: bool,
    axis_len: float,
    max_boxes: int,
    show_normals: bool,
    normals_scale: float,
    max_normals: int,
    orient_normals: bool,
    custom_location_text: str = "",
    custom_orientation_text: str = "",
    target_question_text: str = "",
):
    fig.data = fig.data[:1]
    fig.data[0].marker.size = float(point_size)

    arrow_location = parse_optional_vector(custom_location_text, {3}, "Custom location", strict=False)
    if arrow_location is None:
        arrow_location = payload["location"]

    arrow_orientation = parse_optional_vector(custom_orientation_text, {2, 3, 4}, "Custom orientation", strict=False)
    if arrow_orientation is None:
        arrow_orientation = payload["orientation"]

    if color_mode == "RGB":
        fig.data[0].marker.color = payload["rgb_colors"]
    elif color_mode == "Instance":
        fig.data[0].marker.color = payload["instance_colors"]
    elif color_mode == "Segments":
        fig.data[0].marker.color = payload["segment_colors"]
    elif color_mode == "Normals":
        if orient_normals and arrow_location is not None:
            normals_oriented = orient_normals_toward_viewpoint(
                payload["xyz"], payload["normals"], np.asarray(arrow_location, dtype=np.float32)
            )
            fig.data[0].marker.color = rgb01_to_plotly_strings(normals_to_rgb01(normals_oriented))
        else:
            fig.data[0].marker.color = payload["normal_colors"]
    else:
        fig.data[0].marker.color = payload["rgb_colors"]

    if show_boxes:
        for t in get_bbox_traces(dataset_name, idx, max_points, max_boxes):
            fig.add_trace(copy.deepcopy(t))

    if show_target_box:
        qa = DATA_BY_DATASET[dataset_name][idx]
        question_text = (target_question_text or "").strip() or qa.get("question", "")
        for t in build_target_bbox_traces(
            payload,
            question_text=question_text,
            max_matches=int(target_box_max_matches),
        ):
            fig.add_trace(t)

    if show_axis:
        for t in make_world_axis_traces(payload["center"], axis_len=float(axis_len)):
            fig.add_trace(t)

    if show_arrow and arrow_location is not None and arrow_orientation is not None:
        try:
            for t in make_situation_arrow_trace(
                arrow_location,
                arrow_orientation,
                scale=float(axis_len),
            ):
                fig.add_trace(t)
        except Exception as e:
            print(f"[warn] could not render situation arrow: {e}")

    if show_normals and payload["normals"] is not None:
        viewpoint = None
        if orient_normals and arrow_location is not None:
            viewpoint = np.asarray(arrow_location, dtype=np.float32)

        for t in build_normal_glyph_traces(
            payload["xyz"],
            payload["normals"],
            glyph_scale=float(normals_scale),
            max_glyphs=int(max_normals),
            orient_viewpoint=viewpoint,
        ):
            fig.add_trace(t)

    return fig


# ======================== Split inference ========================

def infer_split_from_scene(dataset_name: str, scan_id: str, qa_idx: int = None) -> str:
    inds = SCAN_TO_INDICES_BY_DATASET[dataset_name].get(scan_id, [])
    splits = sorted({DATA_BY_DATASET[dataset_name][i].get("split", "test") for i in inds})

    if len(splits) == 1:
        return splits[0]

    if qa_idx is not None:
        try:
            return DATA_BY_DATASET[dataset_name][int(qa_idx)].get("split", "test")
        except Exception:
            pass

    for pref in ("test", "val", "train"):
        if pref in splits:
            return pref
    return "test"


def parse_optional_vector(text, expected_lens, field_name: str, *, strict: bool):
    text = (text or "").strip()
    if not text:
        return None

    parts = [p for p in text.replace(",", " ").split() if p]
    try:
        values = [float(p) for p in parts]
    except ValueError as e:
        if strict:
            raise ValueError(f"{field_name} must contain numeric values.")
        return None

    if len(values) not in expected_lens:
        if strict:
            expected = "/".join(str(x) for x in sorted(expected_lens))
            raise ValueError(f"{field_name} must have {expected} values, got {len(values)}.")
        return None

    return values


# ======================== Dropdown callbacks ========================

def scans_for_split(dataset_name: str, split_filter: str):
    data = DATA_BY_DATASET[dataset_name]
    scan_to_inds = SCAN_TO_INDICES_BY_DATASET[dataset_name]

    if split_filter == "all":
        scans = list(scan_to_inds.keys())
    else:
        scans = []
        for sid, inds in scan_to_inds.items():
            if any(data[i].get("split") == split_filter for i in inds):
                scans.append(sid)

    scans = sorted(scans)
    if ONLY_SHOW_SCANS_WITH_PTH:
        scans = [sid for sid in scans if scene_exists(dataset_name, sid)]
    return scans


def on_split_change(dataset_name: str, split_filter: str):
    scans = scans_for_split(dataset_name, split_filter)

    scan_val = None
    qa_choices = []
    question_choices = []
    qa_val = None

    for sid in scans:
        choices = qa_choices_for_scan(dataset_name, sid, split_filter)
        if choices:
            scan_val = sid
            qa_choices = choices
            question_choices = question_choices_for_scan(dataset_name, sid, split_filter)
            qa_val = choices[0][1]
            break

    return (
        gr.update(choices=scans, value=scan_val),
        gr.update(choices=qa_choices, value=qa_val),
        gr.update(choices=question_choices, value=qa_val),
        gr.update(value=question_text_for_idx(dataset_name, qa_val)),
    )


def on_dataset_change(dataset_name: str):
    scans = AVAILABLE_SCANS_BY_DATASET[dataset_name]
    split_val = "all"

    scan_val = None
    qa_choices = []
    question_choices = []
    qa_val = None

    for sid in scans:
        choices = qa_choices_for_scan(dataset_name, sid, split_val)
        if choices:
            scan_val = sid
            qa_choices = choices
            question_choices = question_choices_for_scan(dataset_name, sid, split_val)
            qa_val = choices[0][1]
            break

    return (
        gr.update(choices=scans, value=scan_val),
        gr.update(value=split_val),
        gr.update(choices=qa_choices, value=qa_val),
        gr.update(choices=question_choices, value=qa_val),
        gr.update(value=question_text_for_idx(dataset_name, qa_val)),
    )


def on_scan_change(dataset_name: str, scan_id: str, split_filter: str):
    qa_choices = qa_choices_for_scan(dataset_name, scan_id, split_filter)
    question_choices = question_choices_for_scan(dataset_name, scan_id, split_filter)
    qa_val = qa_choices[0][1] if qa_choices else None
    return (
        gr.update(choices=qa_choices, value=qa_val),
        gr.update(choices=question_choices, value=qa_val),
        gr.update(value=question_text_for_idx(dataset_name, qa_val)),
    )


def on_question_change(dataset_name: str, global_idx):
    return (
        gr.update(value=global_idx),
        gr.update(value=question_text_for_idx(dataset_name, global_idx)),
    )


def on_qa_change(dataset_name: str, global_idx):
    return (
        gr.update(value=global_idx),
        gr.update(value=question_text_for_idx(dataset_name, global_idx)),
    )


# ======================== Render callbacks ========================

def build_base(dataset_name, global_idx, max_points, point_size):
    if global_idx is None:
        raise gr.Error("No QA entry selected.")

    idx = int(global_idx)
    payload, key = get_downsampled_scene(dataset_name, idx, int(max_points))
    fig = make_base_figure(payload, float(point_size))
    return fig, fig, key


def update_style(
    fig,
    key,
    dataset_name,
    global_idx,
    color_mode,
    point_size,
    show_boxes,
    show_target_box,
    target_box_max_matches,
    show_axis,
    show_arrow,
    axis_len,
    max_boxes,
    max_points,
    show_normals,
    normals_scale,
    max_normals,
    orient_normals,
    custom_location_text="",
    custom_orientation_text="",
    target_question_text="",
):
    if global_idx is None:
        return fig, fig, key

    idx = int(global_idx)
    payload, new_key = get_downsampled_scene(dataset_name, idx, int(max_points))

    if fig is None or key != new_key:
        fig = make_base_figure(payload, float(point_size))

    fig = apply_style_and_overlays(
        fig=fig,
        payload=payload,
        dataset_name=dataset_name,
        idx=idx,
        max_points=int(max_points),
        color_mode=color_mode,
        point_size=float(point_size),
        show_boxes=bool(show_boxes),
        show_target_box=bool(show_target_box),
        target_box_max_matches=int(target_box_max_matches),
        show_axis=bool(show_axis),
        show_arrow=bool(show_arrow),
        axis_len=float(axis_len),
        max_boxes=int(max_boxes),
        show_normals=bool(show_normals),
        normals_scale=float(normals_scale),
        max_normals=int(max_normals),
        orient_normals=bool(orient_normals),
        custom_location_text=custom_location_text,
        custom_orientation_text=custom_orientation_text,
        target_question_text=target_question_text,
    )
    return fig, fig, new_key


# ======================== Chat ========================

# def answer_with_model(user_msg: str, dataset_name: str, global_idx, split_value: str):
#     user_msg = (user_msg or "").strip()
#     if not user_msg:
#         return ""

#     if global_idx is None:
#         return "No QA entry selected."

#     if dataset_name != "scannet":
#         return "Model chat is currently wired for ScanNet only (MSQAScanNet). Switch to scannet to ask questions."

#     idx = int(global_idx)
#     qa = DATA_BY_DATASET[dataset_name][idx]
#     scene_id = qa["scan_id"]
#     situation = qa.get("situation", "")

#     scan_id = qa["scan_id"]
#     if split_value == "all":
#         effective_split = infer_split_from_scene(dataset_name, scan_id, qa_idx=global_idx)
#     else:
#         effective_split = split_value

#     try:
#         svc = get_msr3d_service()
#         svc.change_split(effective_split)

#         with MSR3D_LOCK:
#             print(
#                 f"[model] Generating answer for dataset='{dataset_name}', "
#                 f"scan_id='{scan_id}', split='{effective_split}', "
#                 f"situation='{situation}' | user_msg='{user_msg}'"
#             )
#             ans = svc.answer(scene_id=scene_id, question=user_msg, situation=situation)
#         return ans
#     except Exception as e:
#         return f"[error] {type(e).__name__}: {e}"
def answer_with_model(
    user_msg: str,
    dataset_name: str,
    global_idx,
    split_value: str,
    custom_situation_text: str = "",
    uploaded_images=None,
    custom_location_text: str = "",
    custom_orientation_text: str = "",
):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return ""

    if global_idx is None:
        return "No QA entry selected."

    idx = int(global_idx)
    qa = DATA_BY_DATASET[dataset_name][idx]

    scan_id = qa["scan_id"]
    situation = (custom_situation_text or "").strip() or qa.get("situation", "")
    anchor_locs_override = parse_optional_vector(custom_location_text, {3}, "Custom location", strict=True)
    anchor_orientation_override = parse_optional_vector(
        custom_orientation_text, {2, 3, 4}, "Custom orientation", strict=True
    )

    if split_value == "all":
        effective_split = infer_split_from_scene(dataset_name, scan_id, qa_idx=global_idx)
    else:
        effective_split = split_value

    qa_meta = {
        "dataset_name": dataset_name,
        "scan_id": qa["scan_id"],
        "split": effective_split,
        "situation": qa.get("situation", ""),
        "location": qa.get("location", None),
        "orientation": qa.get("orientation", None),
    }
    if anchor_locs_override is not None:
        qa_meta["anchor_locs_override"] = anchor_locs_override
    if anchor_orientation_override is not None:
        qa_meta["anchor_orientation_override"] = anchor_orientation_override

    try:
        svc = get_msr3d_service()
        svc.change_dataset(dataset_name, effective_split)

        with MSR3D_LOCK:
            print(
                f"[model] Generating answer for dataset='{dataset_name}', "
                f"scan_id='{scan_id}', split='{effective_split}', "
                f"situation='{situation}' | user_msg='{user_msg}'"
            )
            ans = svc.answer(
                qa_meta=qa_meta,
                question=user_msg,
                situation=situation,
                images=uploaded_images,
            )
        return ans
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


# def chat_step(user_msg, history, dataset_name, global_idx, split_value):
#     history = history or []
#     user_msg = (user_msg or "").strip()
#     if not user_msg:
#         return "", history

#     history.append({"role": "user", "content": user_msg})
#     model_answer = answer_with_model(user_msg, dataset_name, global_idx, split_value)
#     history.append({"role": "assistant", "content": model_answer})
#     return "", history
def chat_step(
    user_msg,
    history,
    dataset_name,
    global_idx,
    split_value,
    custom_situation_text,
    uploaded_images,
    custom_location_text,
    custom_orientation_text,
):
    history = history or []
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return "", history

    history.append({"role": "user", "content": user_msg})
    model_answer = answer_with_model(
        user_msg,
        dataset_name,
        global_idx,
        split_value,
        custom_situation_text,
        uploaded_images,
        custom_location_text,
        custom_orientation_text,
    )
    history.append({"role": "assistant", "content": model_answer})

    return "", history


def clear_chat():
    return []


# ======================== App ========================

with gr.Blocks(
    css="""
    #main-layout { align-items: stretch; }
    #viz-details, #chat-panel { min-width: 280px; }
    #scene-plot { height: 62vh !important; min-height: 460px; max-height: 640px; }
    #scene-plot > div { height: 100% !important; }
    #chat-panel .wrap { height: 100%; }
    """
) as demo:
    gr.Markdown(
        "## MSQA Multi-Dataset Scene Viewer (Gradio + Plotly)\n"
        "Loader-matching scene format for **ScanNet / ARKit / 3RScan**.\n\n"
        "**scene_fts** is built exactly like your loader scripts:\n"
        "- ScanNet: xyz + rgb(-1,1) + scene_normals, with ScanNet20 remapped segments\n"
        "- 3RScan: xyz + rgb(-1,1) + scene_normals, segments = instance ids\n"
        "- ARKit: xyz + rgb(-1,1) + scene_normals, segments = instance ids\n"
    )

    model_status_md = gr.Markdown(get_model_status_text())

    dataset_dd = gr.Dropdown(
        choices=["scannet", "arkit", "rscan"],
        value="scannet",
        label="Dataset",
        interactive=True,
    )

    with gr.Row():
        scan_id_dd = gr.Dropdown(
            choices=AVAILABLE_SCANS_BY_DATASET["scannet"],
            value=AVAILABLE_SCANS_BY_DATASET["scannet"][0],
            label="Scene (scan_id / folder name)",
            interactive=True,
        )

        split_filter = gr.Dropdown(
            choices=["all", "train", "val", "test"],
            value="all",
            label="Split filter",
            interactive=True,
        )

        qa_dd = gr.Dropdown(
            choices=[],
            label="QA entry (within scene)",
            interactive=True,
        )

        question_dd = gr.Dropdown(
            choices=[],
            label="Question (within scene)",
            interactive=True,
        )

    fig_state = gr.State(None)
    key_state = gr.State(None)

    with gr.Row(elem_id="main-layout"):
        with gr.Column(scale=3, min_width=280, elem_id="viz-details"):
            gr.Markdown("### Visualization details")

            color_mode = gr.Dropdown(
                choices=["RGB", "Instance", "Segments", "Normals"],
                value="RGB",
                label="Color mode",
            )
            point_size = gr.Slider(1, 10, value=2, step=1, label="Point size")
            max_points = gr.Slider(10_000, 500_000, value=200_000, step=10_000, label="Max points")

            show_boxes = gr.Checkbox(value=False, label="Show instance bounding boxes")
            max_boxes = gr.Slider(10, 500, value=200, step=10, label="Max boxes")

            show_target_box = gr.Checkbox(value=True, label="Show target object bounding box")
            target_box_max_matches = gr.Slider(1, 20, value=8, step=1, label="Max target matches")

            show_axis = gr.Checkbox(value=False, label="Show world axis")
            show_arrow = gr.Checkbox(value=True, label="Show situation arrow")
            axis_len = gr.Slider(0.5, 5.0, value=1.5, step=0.1, label="Axis / arrow scale")

            show_normals = gr.Checkbox(value=False, label="Show normals glyphs")
            orient_normals = gr.Checkbox(value=False, label="Orient normals toward viewpoint")
            normals_scale = gr.Slider(0.02, 0.5, value=0.12, step=0.01, label="Normals glyph scale")
            max_normals = gr.Slider(100, 10000, value=2500, step=100, label="Max normals glyphs")

            btn = gr.Button("Render")

        with gr.Column(scale=7, min_width=520):
            plot = gr.Plot(elem_id="scene-plot", scale=5)

        with gr.Column(scale=3, min_width=300, elem_id="chat-panel"):
            gr.Markdown("### Ask your model about the scene")
            chat = gr.Chatbot(label="Dialogue", height=360)
            with gr.Accordion("Optional Situation Override", open=False):
                custom_situation = gr.Textbox(
                    label="Custom situation override",
                    placeholder="Optional: describe your own situation. Leave empty to use the selected QA situation.",
                    lines=3,
                )
            with gr.Accordion("Optional Multimodal / Pose Overrides", open=False):
                gr.Markdown(
                    "Uploaded images are passed to the model as reference images.\n\n"
                    "If you also want the spatial anchor and scene arrow to change, provide custom pose values below."
                )
                uploaded_images = gr.File(
                    label="Reference images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                custom_location = gr.Textbox(
                    label="Custom location xyz",
                    placeholder="e.g. 1.2, -0.5, 1.6",
                    lines=1,
                )
                custom_orientation = gr.Textbox(
                    label="Custom orientation xyzw or facing vector xy[z]",
                    placeholder="e.g. 0, 0, 0.707, 0.707",
                    lines=1,
                )
            user_msg = gr.Textbox(label="Ask a question", placeholder="Ask about the scene...", lines=2)
            with gr.Row():
                send = gr.Button("Send")
                clear = gr.Button("Clear")

    # Init
    demo.load(
        fn=on_dataset_change,
        inputs=[dataset_dd],
        outputs=[scan_id_dd, split_filter, qa_dd, question_dd, user_msg],
    ).then(
        fn=build_base,
        inputs=[dataset_dd, qa_dd, max_points, point_size],
        outputs=[plot, fig_state, key_state],
    ).then(
        fn=update_style,
        inputs=[
            fig_state, key_state, dataset_dd, qa_dd,
            color_mode, point_size, show_boxes, show_target_box,
            target_box_max_matches, show_axis, show_arrow,
            axis_len, max_boxes, max_points,
            show_normals, normals_scale, max_normals, orient_normals,
            custom_location, custom_orientation, user_msg,
        ],
        outputs=[plot, fig_state, key_state],
    ).then(
        fn=get_model_status_text,
        inputs=[],
        outputs=[model_status_md],
    )

    # Dropdown changes
    dataset_dd.change(
        fn=on_dataset_change,
        inputs=[dataset_dd],
        outputs=[scan_id_dd, split_filter, qa_dd, question_dd, user_msg],
    )
    split_filter.change(
        fn=on_split_change,
        inputs=[dataset_dd, split_filter],
        outputs=[scan_id_dd, qa_dd, question_dd, user_msg],
    )
    scan_id_dd.change(
        fn=on_scan_change,
        inputs=[dataset_dd, scan_id_dd, split_filter],
        outputs=[qa_dd, question_dd, user_msg],
    )
    question_dd.change(
        fn=on_question_change,
        inputs=[dataset_dd, question_dd],
        outputs=[qa_dd, user_msg],
    ).then(
        fn=update_style,
        inputs=[
            fig_state, key_state, dataset_dd, qa_dd,
            color_mode, point_size, show_boxes, show_target_box,
            target_box_max_matches, show_axis, show_arrow,
            axis_len, max_boxes, max_points,
            show_normals, normals_scale, max_normals, orient_normals,
            custom_location, custom_orientation, user_msg,
        ],
        outputs=[plot, fig_state, key_state],
    )
    qa_dd.change(
        fn=on_qa_change,
        inputs=[dataset_dd, qa_dd],
        outputs=[question_dd, user_msg],
    ).then(
        fn=update_style,
        inputs=[
            fig_state, key_state, dataset_dd, qa_dd,
            color_mode, point_size, show_boxes, show_target_box,
            target_box_max_matches, show_axis, show_arrow,
            axis_len, max_boxes, max_points,
            show_normals, normals_scale, max_normals, orient_normals,
            custom_location, custom_orientation, user_msg,
        ],
        outputs=[plot, fig_state, key_state],
    )

    # Heavy render
    btn.click(
        fn=build_base,
        inputs=[dataset_dd, qa_dd, max_points, point_size],
        outputs=[plot, fig_state, key_state],
    ).then(
        fn=update_style,
        inputs=[
            fig_state, key_state, dataset_dd, qa_dd,
            color_mode, point_size, show_boxes, show_target_box,
            target_box_max_matches, show_axis, show_arrow,
            axis_len, max_boxes, max_points,
            show_normals, normals_scale, max_normals, orient_normals,
            custom_location, custom_orientation, user_msg,
        ],
        outputs=[plot, fig_state, key_state],
    )

    # Light updates
    for comp in [
        color_mode, point_size, show_boxes, show_target_box, target_box_max_matches,
        show_axis, show_arrow, axis_len, max_boxes,
        show_normals, normals_scale, max_normals, orient_normals, custom_location, custom_orientation, user_msg
    ]:
        comp.change(
            fn=update_style,
            inputs=[
                fig_state, key_state, dataset_dd, qa_dd,
                color_mode, point_size, show_boxes, show_target_box,
                target_box_max_matches, show_axis, show_arrow,
                axis_len, max_boxes, max_points,
                show_normals, normals_scale, max_normals, orient_normals,
                custom_location, custom_orientation, user_msg,
            ],
            outputs=[plot, fig_state, key_state],
        )

    # max_points is heavy
    max_points.change(
        fn=build_base,
        inputs=[dataset_dd, qa_dd, max_points, point_size],
        outputs=[plot, fig_state, key_state],
    ).then(
        fn=update_style,
        inputs=[
            fig_state, key_state, dataset_dd, qa_dd,
            color_mode, point_size, show_boxes, show_target_box,
            target_box_max_matches, show_axis, show_arrow,
            axis_len, max_boxes, max_points,
            show_normals, normals_scale, max_normals, orient_normals,
            custom_location, custom_orientation, user_msg,
        ],
        outputs=[plot, fig_state, key_state],
    )

    # Chat
    send.click(
        fn=chat_step,
        inputs=[
            user_msg, chat, dataset_dd, qa_dd, split_filter,
            custom_situation, uploaded_images, custom_location, custom_orientation,
        ],
        outputs=[user_msg, chat],
    )
    user_msg.submit(
        fn=chat_step,
        inputs=[
            user_msg, chat, dataset_dd, qa_dd, split_filter,
            custom_situation, uploaded_images, custom_location, custom_orientation,
        ],
        outputs=[user_msg, chat],
    )
    clear.click(fn=clear_chat, inputs=[], outputs=[chat])


if __name__ == "__main__":
    warmup_model()
    demo.launch(share=True)
