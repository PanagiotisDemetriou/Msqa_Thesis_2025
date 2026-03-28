#!/usr/bin/env bash
#
# Scene normals viewer runner (similar to your object runner)
#

# =========================
#  WSL / X11 Configuration
# =========================
if [[ -n "${WSL_DISTRO_NAME-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
  echo "[info] Detected WSL environment."
  export GDK_BACKEND=x11
  export QT_QPA_PLATFORM=xcb
  export XDG_SESSION_TYPE=x11

  if [[ -z "${DISPLAY-}" ]]; then
    export DISPLAY=:0
  fi
fi

# =========================
#  Defaults
# =========================
PYTHON_SCRIPT="msr3d/tools/scene_normals_viewer.py"
PYTHON_BIN="python3"
MODE_CHOICE=""
SCAN_ID="scene0000_00"
SPLIT="train"
CFG="msr3d/configs/data.yaml"
#NORMALS_DIR="/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals/"
#NORMALS_DIR="/mnt/d/Thesis/data/MSR3D_v2_pcds/ARkit_base/scan_data/pcd_normals/"
NORMALS_DIR="/mnt/d/Thesis/data/MSR3D_v2_pcds/rscan_base/scan_data/pcd_normals/"
VOXEL="0.02"
POINT_SIZE="2.0"
COLOR_BY_NORMALS="1"   # enabled by default for comparison
KEEP_SCENE_RGB="0"
NO_NORMALS="0"
ORIENT_VIEWPOINT=""    # e.g. "0 0 0"

export PYTHONPATH="$PWD:$PWD/msr3d:$PWD/Pointcept_main:$PYTHONPATH"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_help() {
  cat <<EOF
Usage: $0 [options]

Options:
  -s <script>         Python script path (default: ${PYTHON_SCRIPT})
  -p <python>         Python executable (default: ${PYTHON_BIN})
  -m <mode>           Mode: "wsl" or "normal" (auto-detect if omitted)
  --scan <id>         Scan id (default: ${SCAN_ID})
  --split <name>      Split: train/val/test (default: ${SPLIT})
  --cfg <path>        Config yaml path (default: ${CFG})
  --normals <dir>     Normals cache directory (default: ${NORMALS_DIR})
  --voxel <v>         Voxel size for downsampling (default: ${VOXEL})
  --psize <p>         Point size (default: ${POINT_SIZE})
  --no_normals        Do not show normal glyphs
  --no_color_normals  Do not color by normals (use RGB if available)
  --keep_scene_rgb    If set, keep scene RGB even if coloring by normals
  --orient "x y z"    Flip normals toward viewpoint (reduces sign flip), e.g. --orient "0 0 0"
  -h                  Show this help

Examples:
  $0 --scan scene0000_00
  $0 --scan scene0000_00 --voxel 0.02 --orient "0 0 0"
  $0 --scan scene0000_00 --no_color_normals
EOF
}

# =========================
#  Parse arguments
# =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s) PYTHON_SCRIPT="$2"; shift 2;;
    -p) PYTHON_BIN="$2"; shift 2;;
    -m) MODE_CHOICE="$2"; shift 2;;
    --scan) SCAN_ID="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --cfg) CFG="$2"; shift 2;;
    --normals) NORMALS_DIR="$2"; shift 2;;
    --voxel) VOXEL="$2"; shift 2;;
    --psize) POINT_SIZE="$2"; shift 2;;
    --no_normals) NO_NORMALS="1"; shift 1;;
    --no_color_normals) COLOR_BY_NORMALS="0"; shift 1;;
    --keep_scene_rgb) KEEP_SCENE_RGB="1"; shift 1;;
    --orient) ORIENT_VIEWPOINT="$2"; shift 2;;
    -h) print_help; exit 0;;
    *) echo -e "${RED}Unknown option: $1${NC}"; print_help; exit 1;;
  esac
done

echo -e "${GREEN}=== Scene Normals Viewer (Cache) ===${NC}\n"

# Auto-detect MODE
if [[ -z "${MODE_CHOICE}" ]]; then
  MODE_CHOICE="normal"
  if [[ -n "${WSL_DISTRO_NAME-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
    MODE_CHOICE="wsl"
    echo -e "${YELLOW}[auto-detect] Running in WSL mode${NC}"
  else
    echo -e "${YELLOW}[auto-detect] Running in normal mode${NC}"
  fi
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo -e "${RED}Error: Python script '$PYTHON_SCRIPT' not found!${NC}"
  exit 1
fi

echo -e "${GREEN}✓ Using Python:${NC} ${BLUE}${PYTHON_BIN}${NC}"
echo -e "${GREEN}✓ Using script:${NC} ${BLUE}${PYTHON_SCRIPT}${NC}"
echo -e "${GREEN}✓ scan_id:${NC} ${YELLOW}${SCAN_ID}${NC}"
echo -e "${GREEN}✓ normals_dir:${NC} ${BLUE}${NORMALS_DIR}${NC}"
echo -e "${GREEN}✓ voxel:${NC} ${YELLOW}${VOXEL}${NC}  ${GREEN}point_size:${NC} ${YELLOW}${POINT_SIZE}${NC}\n"

ARGS=(
  --cfg "$CFG" --split "$SPLIT"
  --scan_id "$SCAN_ID"
  --normals_dir "$NORMALS_DIR"
  --voxel "$VOXEL" --point_size "$POINT_SIZE" --normal_colors --save_obj
)

if [[ "$NO_NORMALS" == "1" ]]; then
  ARGS+=(--no_normals)
fi

if [[ "$COLOR_BY_NORMALS" == "1" ]]; then
  ARGS+=(--color_by_normals)
fi

if [[ "$KEEP_SCENE_RGB" == "1" ]]; then
  ARGS+=(--keep_scene_rgb)
fi

if [[ -n "$ORIENT_VIEWPOINT" ]]; then
  # shellcheck disable=SC2206
  ORIENT_ARR=($ORIENT_VIEWPOINT)
  if [[ ${#ORIENT_ARR[@]} -ne 3 ]]; then
    echo -e "${RED}Error: --orient expects exactly 3 numbers, got: '${ORIENT_VIEWPOINT}'${NC}"
    exit 1
  fi
  ARGS+=(--orient_viewpoint "${ORIENT_ARR[0]}" "${ORIENT_ARR[1]}" "${ORIENT_ARR[2]}")
fi

echo -e "${YELLOW}Close the Open3D window (Q / ESC) to return to terminal.${NC}\n"

# Run
if [[ "$MODE_CHOICE" == "wsl" ]]; then
  XDG_SESSION_TYPE=x11 "$PYTHON_BIN" "$PYTHON_SCRIPT" "${ARGS[@]}"
else
  "$PYTHON_BIN" "$PYTHON_SCRIPT" "${ARGS[@]}"
fi

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo -e "${RED}Python script exited with code ${EXIT_CODE}.${NC}"
else
  echo -e "${GREEN}Python script completed successfully.${NC}"
fi
exit $EXIT_CODE
