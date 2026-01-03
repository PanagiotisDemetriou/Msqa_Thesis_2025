#!/bin/bash

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
PYTHON_SCRIPT="msr3d/tools/object_normals_viewer.py"
PYTHON_BIN="python3"
MODE_CHOICE=""
SCAN_ID="scene0000_00"
OBJ_IDX="0"
SPLIT="train"
CFG="msr3d/configs/data.yaml"
NORMALS_DIR="/mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_normals/"
VOXEL="0.0"
POINT_SIZE="3.0"
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
  -s <script>      Python script path (default: ${PYTHON_SCRIPT})
  -p <python>      Python executable (default: ${PYTHON_BIN})
  -m <mode>        Mode: "wsl" or "normal" (auto-detect if omitted)
  --scan <id>      Scan id (default: ${SCAN_ID})
  --obj <idx>      Object index (default: ${OBJ_IDX})
  --split <name>   Split: train/val/test (default: ${SPLIT})
  --cfg <path>     Config yaml path (default: ${CFG})
  --normals <dir>  Normals cache directory (default: ${NORMALS_DIR})
  --voxel <v>      Voxel size for downsampling (default: ${VOXEL})
  --psize <p>      Point size (default: ${POINT_SIZE})
  -h               Show this help

Examples:
  $0 --scan scene0000_00 --obj 0
  $0 --scan scene0000_00 --obj 5 --voxel 0.02
  $0 -p /home/panagiotis/miniconda3/envs/pointcept-torch2.5.0-cu12.4/bin/python \\
     --scan scene0000_00 --obj 0
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
    --obj) OBJ_IDX="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --cfg) CFG="$2"; shift 2;;
    --normals) NORMALS_DIR="$2"; shift 2;;
    --voxel) VOXEL="$2"; shift 2;;
    --psize) POINT_SIZE="$2"; shift 2;;
    -h) print_help; exit 0;;
    *) echo -e "${RED}Unknown option: $1${NC}"; print_help; exit 1;;
  esac
done

echo -e "${GREEN}=== Object Normals Viewer ===${NC}\n"

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
echo -e "${GREEN}✓ scan_id:${NC} ${YELLOW}${SCAN_ID}${NC}  ${GREEN}obj_idx:${NC} ${YELLOW}${OBJ_IDX}${NC}"
echo -e "${GREEN}✓ normals_dir:${NC} ${BLUE}${NORMALS_DIR}${NC}"
echo -e "${GREEN}✓ voxel:${NC} ${YELLOW}${VOXEL}${NC}  ${GREEN}point_size:${NC} ${YELLOW}${POINT_SIZE}${NC}\n"

echo -e "${YELLOW}Close the Open3D window (Q / ESC) to return to terminal.${NC}\n"

# Run
if [[ "$MODE_CHOICE" == "wsl" ]]; then
  XDG_SESSION_TYPE=x11 "$PYTHON_BIN" "$PYTHON_SCRIPT" \
    --cfg "$CFG" --split "$SPLIT" \
    --scan_id "$SCAN_ID" --obj_idx "$OBJ_IDX" \
    --normals_dir "$NORMALS_DIR" \
    --voxel "$VOXEL" --point_size "$POINT_SIZE"
else
  "$PYTHON_BIN" "$PYTHON_SCRIPT" \
    --cfg "$CFG" --split "$SPLIT" \
    --scan_id "$SCAN_ID" --obj_idx "$OBJ_IDX" \
    --normals_dir "$NORMALS_DIR" \
    --voxel "$VOXEL" --point_size "$POINT_SIZE"
fi

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo -e "${RED}Python script exited with code ${EXIT_CODE}.${NC}"
else
  echo -e "${GREEN}Python script completed successfully.${NC}"
fi
exit $EXIT_CODE
