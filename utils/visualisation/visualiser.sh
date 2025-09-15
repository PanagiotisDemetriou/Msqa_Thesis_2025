#!/usr/bin/env bash
set -euo pipefail

# Defaults
PYTHON_SCRIPT="scannet_scene_visualiser.py"   # default simple visualiser
SCENE_DIR="./scannet/test/scene0707_00"
DOWNSAMPLE="0.01"
OUT_PLY="scene0707_00.ply"
MODE_CHOICE=""        # "wsl" or "normal"
VIS_CHOICE=""         # "simple" or "advanced"

print_help() {
  cat <<EOF
Usage: $0 [options]

Options:
  -s <script>      Python script path (default: ${PYTHON_SCRIPT})
  -i <scene_dir>   Scene directory (default: ${SCENE_DIR})
  -v <voxel>       Downsample voxel size in meters (default: ${DOWNSAMPLE})
  -o <outfile>     Output PLY path (default: ${OUT_PLY})
  -m <mode>        Mode: "wsl" or "normal" (if omitted, you'll be prompted)
  -V <visualiser>  Visualiser: "simple" or "advanced" (if omitted, you'll be prompted)
  -h               Show this help

Examples:
  $0
  $0 -m wsl
  $0 -V advanced
  $0 -s view_scannet_npy.py -i ./scannet/test/scene0707_00 -v 0.02 -o out.ply
EOF
}

while getopts ":s:i:v:o:m:V:h" opt; do
  case "$opt" in
    s) PYTHON_SCRIPT="$OPTARG" ;;
    i) SCENE_DIR="$OPTARG" ;;
    v) DOWNSAMPLE="$OPTARG" ;;
    o) OUT_PLY="$OPTARG" ;;
    m) MODE_CHOICE="$OPTARG" ;;
    V) VIS_CHOICE="$OPTARG" ;;
    h) print_help; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; print_help; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# Sanity checks on scene/script if provided explicitly
if [[ -n "${PYTHON_SCRIPT}" && ! -f "$PYTHON_SCRIPT" ]]; then
  # We will possibly override script based on VIS_CHOICE below, so only warn here
  echo "[warn] Provided script not found yet: $PYTHON_SCRIPT (may be changed by visualiser choice)" >&2
fi

# -------- Determine MODE (normal vs WSL) --------
if [[ -z "${MODE_CHOICE}" ]]; then
  DEFAULT_MODE="normal"
  if [[ -n "${WSL_DISTRO_NAME-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
    DEFAULT_MODE="wsl"
  fi
  echo "Choose run mode:"
  echo "  1) WSL"
  echo "  2) Normal"
  read -rp "Select 1 or 2 [default: ${DEFAULT_MODE}]: " pick
  case "${pick:-$DEFAULT_MODE}" in
    1|wsl|WSL) MODE_CHOICE="wsl" ;;
    2|normal|NORMAL) MODE_CHOICE="normal" ;;
    *) MODE_CHOICE="$DEFAULT_MODE" ;;
  esac
fi

# -------- Determine visualiser (simple vs advanced) --------
if [[ -z "${VIS_CHOICE}" ]]; then
  DEFAULT_VIS="advanced"
  echo "Choose visualiser:"
  echo "  1) simple"
  echo "  2) advanced"
  read -rp "Select 1 or 2 [default: ${DEFAULT_VIS}]: " pick
  case "${pick:-$DEFAULT_VIS}" in
    1|simple|SIMPLE) VIS_CHOICE="simple" ;;
    2|advanced|ADVANCED) VIS_CHOICE="advanced" ;;
    *) VIS_CHOICE="$DEFAULT_VIS" ;;
  esac
fi

# -------- Select script by visualiser (unless user explicitly set a custom one) --------
if [[ "$VIS_CHOICE" == "simple" ]]; then
  if [[ "$PYTHON_SCRIPT" == "scannet_scene_visualiser.py" || ! -f "$PYTHON_SCRIPT" ]]; then
    PYTHON_SCRIPT="utils/visualisation/scannet_scene_visualiser.py"
  fi
else
  # advanced viewer (VisualizerWithKeyCallback)
  if [[ "$PYTHON_SCRIPT" == "scannet_scene_visualiser.py" || ! -f "$PYTHON_SCRIPT" ]]; then
    PYTHON_SCRIPT="utils/visualisation/scannet_advanced_visualiser.py"
  fi
fi

# Final sanity check
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "[error] Python script not found: $PYTHON_SCRIPT" >&2
  exit 1
fi

# Build command
CMD=( python "$PYTHON_SCRIPT" "$SCENE_DIR")

echo "[info] Script: $PYTHON_SCRIPT"
echo "[info] Scene:  $SCENE_DIR"
echo "[info] Down:   $DOWNSAMPLE m"
echo "[info] Output: $OUT_PLY"
echo "[info] Mode:   $MODE_CHOICE"
echo "[info] Visualisation:    $VIS_CHOICE"

# Run
if [[ "$MODE_CHOICE" == "wsl" ]]; then
  echo "[info] Running with XDG_SESSION_TYPE=x11"
  XDG_SESSION_TYPE=x11 "${CMD[@]}"
else
  "${CMD[@]}"
fi

#  ./utils/visualisation/visualiser.sh -i data/scannet/train/scene0000_00