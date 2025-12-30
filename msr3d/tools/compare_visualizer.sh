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

PYTHON_SCRIPT="msr3d/tools/compare_data_formats_visualizer.py"
PYTHON_BIN="python3"
MODE_CHOICE=""

OLD_PATH=""
NEW_PATH=""
OBJ_IDX=""
POINT_SIZE="3.0"
GAP="2.0"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_help() {
  cat <<EOF
Usage: $0 [options] --old <old_scene.pth> --new <new_obj_pcds.pth> --obj_idx <int>

Required:
  --old <path>        Old scene .pth (points, colors, instance_labels)
  --new <path>        New obj_pcds .pth (tensor (1,N,P,6) or (N,P,6))
  --obj_idx <int>     Object index / instance id to compare

Options:
  -s <script>         Python script path (default: ${PYTHON_SCRIPT})
  -p <python>         Python executable (default: ${PYTHON_BIN})
  -m <mode>           Mode: "wsl" or "normal" (auto-detect if omitted)
  --point_size <f>    Point size (default: ${POINT_SIZE})
  --gap <f>           Spacing multiplier (default: ${GAP})
  -h                  Help

Example:
  $0 -p /home/panagiotis/miniconda3/envs/pointcept-torch2.5.0-cu12.4/bin/python \\
     --old /mnt/d/.../scene0000_00.pth \\
     --new /home/panagiotis/msqa/Msqa_Thesis_2025/scene0000_00_new_obj_pcds.pth \\
     --obj_idx 10
EOF
}

# =========================
#  Parse args
# =========================

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s) PYTHON_SCRIPT="$2"; shift 2 ;;
    -p) PYTHON_BIN="$2"; shift 2 ;;
    -m) MODE_CHOICE="$2"; shift 2 ;;
    --old) OLD_PATH="$2"; shift 2 ;;
    --new) NEW_PATH="$2"; shift 2 ;;
    --obj_idx) OBJ_IDX="$2"; shift 2 ;;
    --point_size) POINT_SIZE="$2"; shift 2 ;;
    --gap) GAP="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    *)
      echo -e "${RED}Unknown argument: $1${NC}" >&2
      print_help
      exit 1
      ;;
  esac
done

echo -e "${GREEN}=== Compare One Object: OLD vs NEW ===${NC}\n"

if [[ -z "${MODE_CHOICE}" ]]; then
  MODE_CHOICE="normal"
  if [[ -n "${WSL_DISTRO_NAME-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
    MODE_CHOICE="wsl"
    echo -e "${YELLOW}[auto-detect] Running in WSL mode${NC}"
  else
    echo -e "${YELLOW}[auto-detect] Running in normal mode${NC}"
  fi
fi

if [[ -z "${OLD_PATH}" ]] || [[ -z "${NEW_PATH}" ]] || [[ -z "${OBJ_IDX}" ]]; then
  echo -e "${RED}Error: --old, --new and --obj_idx are required.${NC}"
  print_help
  exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo -e "${RED}Error: Python script '$PYTHON_SCRIPT' not found!${NC}"
  exit 1
fi
if [ ! -f "$OLD_PATH" ]; then
  echo -e "${RED}Error: OLD file not found: ${OLD_PATH}${NC}"
  exit 1
fi
if [ ! -f "$NEW_PATH" ]; then
  echo -e "${RED}Error: NEW file not found: ${NEW_PATH}${NC}"
  exit 1
fi

echo -e "${GREEN}✓ Python:${NC} ${BLUE}${PYTHON_BIN}${NC}"
echo -e "${GREEN}✓ Script:${NC} ${BLUE}${PYTHON_SCRIPT}${NC}"
echo -e "${GREEN}✓ Mode:${NC} ${YELLOW}${MODE_CHOICE}${NC}"
echo -e "${GREEN}✓ OLD:${NC} ${BLUE}${OLD_PATH}${NC}"
echo -e "${GREEN}✓ NEW:${NC} ${BLUE}${NEW_PATH}${NC}"
echo -e "${GREEN}✓ obj_idx:${NC} ${YELLOW}${OBJ_IDX}${NC}"
echo -e "${GREEN}✓ point_size:${NC} ${YELLOW}${POINT_SIZE}${NC}"
echo -e "${GREEN}✓ gap:${NC} ${YELLOW}${GAP}${NC}\n"

CMD=( "$PYTHON_BIN" "$PYTHON_SCRIPT"
  --old "$OLD_PATH"
  --new "$NEW_PATH"
  --obj_idx "$OBJ_IDX"
  --point_size "$POINT_SIZE"
  --gap "$GAP"
)

if [[ "$MODE_CHOICE" == "wsl" ]]; then
  XDG_SESSION_TYPE=x11 "${CMD[@]}"
else
  "${CMD[@]}"
fi

exit $?
# run
#./msr3d/tools/compare_visualizer.sh   -p /home/panagiotis/miniconda3/envs/pointcept-torch2.5.0-cu12.4/bin/python   --old /mnt/d/Thesis/data/MSR3D_v2_pcds/scannet_base/scan_data/pcd_with_global_alignment/scene0000_00.pth   --new /home/panagiotis/msqa/Msqa_Thesis_2025/msr3d/tools/scene0000_00_new_obj_pcds.pth   --obj_idx 30