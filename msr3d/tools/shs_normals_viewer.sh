# #!/bin/bash
# set -e

# # =========================
# #  WSL / X11 Configuration
# # =========================
# if [[ -n "${WSL_DISTRO_NAME-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
#   echo "[info] Detected WSL environment."
#   export GDK_BACKEND=x11
#   export QT_QPA_PLATFORM=xcb
#   export XDG_SESSION_TYPE=x11
#   if [[ -z "${DISPLAY-}" ]]; then
#     export DISPLAY=:0
#   fi
# fi

# # =========================
# #  Defaults
# # =========================
# PYTHON_SCRIPT="msr3d/tools/shs_net_normals_viewer.py"
# PYTHON_BIN="python3"
# MODE_CHOICE=""

# SCAN_ID="scene0000_00"
# SPLIT="train"
# CFG="msr3d/configs/data.yaml"

# NORMALS_DIR="/home/panagiotis/msqa/Msqa_Thesis_2025/SHS-Net/log/001/results_Scannet/ckpt_800/pred_normal/"
# NORMALS_PATH=""

# VOXEL="0.0"
# POINT_SIZE="2.0"
# NO_NORMALS_FLAG=""

# export PYTHONPATH="$PWD:$PWD/msr3d:$PWD/Pointcept_main:$PYTHONPATH"

# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m'

# print_help() {
#   cat <<EOF
# Usage: $0 [options]

# Options:
#   -s <script>           Python script path (default: ${PYTHON_SCRIPT})
#   -p <python>           Python executable (default: ${PYTHON_BIN})
#   -m <mode>             Mode: "wsl" or "normal" (auto-detect if omitted)

#   --scan <id>           Scan id (default: ${SCAN_ID})
#   --split <name>        Split: train/val/test (default: ${SPLIT})
#   --cfg <path>          Config yaml path (default: ${CFG})

#   --normals_dir <dir>   Directory of scene .normals files (default: ${NORMALS_DIR})
#   --normals_path <p>    Explicit .normals file path (overrides --normals_dir)

#   --voxel <v>           Voxel size for downsampling (default: ${VOXEL})
#   --psize <p>           Point size (default: ${POINT_SIZE})
#   --no_normals          Do not show normals

#   -h                    Show this help

# Examples:
#   $0 --scan scene0000_00
#   $0 --scan scene0000_00 --voxel 0.02
#   $0 --scan scene0000_00 --normals_path /path/to/scene0000_00.normals
# EOF
# }

# # =========================
# #  Parse arguments
# # =========================
# while [[ $# -gt 0 ]]; do
#   case "$1" in
#     -s) PYTHON_SCRIPT="$2"; shift 2;;
#     -p) PYTHON_BIN="$2"; shift 2;;
#     -m) MODE_CHOICE="$2"; shift 2;;

#     --scan) SCAN_ID="$2"; shift 2;;
#     --split) SPLIT="$2"; shift 2;;
#     --cfg) CFG="$2"; shift 2;;

#     --normals_dir) NORMALS_DIR="$2"; shift 2;;
#     --normals_path) NORMALS_PATH="$2"; shift 2;;

#     --voxel) VOXEL="$2"; shift 2;;
#     --psize) POINT_SIZE="$2"; shift 2;;
#     --no_normals) NO_NORMALS_FLAG="--no_normals"; shift 1;;

#     -h) print_help; exit 0;;
#     *) echo -e "${RED}Unknown option: $1${NC}"; print_help; exit 1;;
#   esac
# done

# echo -e "${GREEN}=== Scene Normals Viewer (.normals) ===${NC}\n"

# # Auto-detect MODE
# if [[ -z "${MODE_CHOICE}" ]]; then
#   MODE_CHOICE="normal"
#   if [[ -n "${WSL_DISTRO_NAME-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
#     MODE_CHOICE="wsl"
#     echo -e "${YELLOW}[auto-detect] Running in WSL mode${NC}"
#   else
#     echo -e "${YELLOW}[auto-detect] Running in normal mode${NC}"
#   fi
# fi

# if [[ ! -f "$PYTHON_SCRIPT" ]]; then
#   echo -e "${RED}Error: Python script '$PYTHON_SCRIPT' not found!${NC}"
#   exit 1
# fi

# echo -e "${GREEN}✓ Using Python:${NC} ${BLUE}${PYTHON_BIN}${NC}"
# echo -e "${GREEN}✓ Using script:${NC} ${BLUE}${PYTHON_SCRIPT}${NC}"
# echo -e "${GREEN}✓ scan_id:${NC} ${YELLOW}${SCAN_ID}${NC}"
# echo -e "${GREEN}✓ normals_dir:${NC} ${BLUE}${NORMALS_DIR}${NC}"
# if [[ -n "$NORMALS_PATH" ]]; then
#   echo -e "${GREEN}✓ normals_path:${NC} ${BLUE}${NORMALS_PATH}${NC}"
# fi
# echo -e "${GREEN}✓ voxel:${NC} ${YELLOW}${VOXEL}${NC}  ${GREEN}point_size:${NC} ${YELLOW}${POINT_SIZE}${NC}\n"

# echo -e "${YELLOW}Close the Open3D window (Q / ESC) to return to terminal.${NC}\n"

# ARGS=( --cfg "$CFG" --split "$SPLIT"
#        --scan_id "$SCAN_ID"
#        --normals_dir "$NORMALS_DIR"
#        --voxel "$VOXEL" --point_size "$POINT_SIZE" )

# if [[ -n "$NORMALS_PATH" ]]; then
#   ARGS+=( --normals_path "$NORMALS_PATH" )
# fi
# if [[ -n "$NO_NORMALS_FLAG" ]]; then
#   ARGS+=( "$NO_NORMALS_FLAG" )
# fi

# if [[ "$MODE_CHOICE" == "wsl" ]]; then
#   XDG_SESSION_TYPE=x11 "$PYTHON_BIN" "$PYTHON_SCRIPT" "${ARGS[@]}"
# else
#   "$PYTHON_BIN" "$PYTHON_SCRIPT" "${ARGS[@]}"
# fi

# echo -e "${GREEN}Python script completed successfully.${NC}"
#!/bin/bash
set -e

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
PYTHON_SCRIPT="msr3d/tools/shs_net_normals_viewer.py"
PYTHON_BIN="python3"
MODE_CHOICE=""

SCAN_ID="scene0000_00"
SPLIT="train"
CFG="msr3d/configs/data.yaml"

# Viewer-specific (new)
VIEW_MODE="pth"     # pth | scene | scene_nn
PC_PTH=""           # required for pth and scene_nn

NORMALS_DIR="/home/panagiotis/msqa/Msqa_Thesis_2025/SHS-Net/log/001/results_Scannet/ckpt_800/pred_normal/"
NORMALS_PATH=""

VOXEL="0.0"
POINT_SIZE="2.0"
NO_NORMALS_FLAG=""

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
  -s <script>           Python script path (default: ${PYTHON_SCRIPT})
  -p <python>           Python executable (default: ${PYTHON_BIN})
  -m <mode>             Runner mode: "wsl" or "normal" (auto-detect if omitted)

  --scan <id>           Scan id (default: ${SCAN_ID})
  --split <name>        Split: train/val/test (default: ${SPLIT})
  --cfg <path>          Config yaml path (default: ${CFG})

  --normals_dir <dir>   Directory of predicted normals files (default: ${NORMALS_DIR})
  --normals_path <p>    Explicit normals file path (overrides --normals_dir)

  --mode <m>            Viewer mode: pth | scene | scene_nn (default: ${VIEW_MODE})
  --pc_pth <path>       Path to SHS-Net input .pth for this scan (required for pth/scene_nn)

  --voxel <v>           Voxel size for downsampling (default: ${VOXEL})
  --psize <p>           Point size (default: ${POINT_SIZE})
  --no_normals          Do not show normals

  -h                    Show this help

Examples:
  # Recommended (visualize on SHS-Net point set)
  $0 --scan scene0000_00 --mode pth --pc_pth /path/to/scene0000_00.pth

  # Dense scene points only (will show points if mismatch)
  $0 --scan scene0000_00 --mode scene

  # Dense scene with NN-projected normals (requires --pc_pth; can be heavy)
  $0 --scan scene0000_00 --mode scene_nn --pc_pth /path/to/scene0000_00.pth --voxel 0.03
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

    --normals_dir) NORMALS_DIR="$2"; shift 2;;
    --normals_path) NORMALS_PATH="$2"; shift 2;;

    --mode) VIEW_MODE="$2"; shift 2;;
    --pc_pth) PC_PTH="$2"; shift 2;;

    --voxel) VOXEL="$2"; shift 2;;
    --psize) POINT_SIZE="$2"; shift 2;;
    --no_normals) NO_NORMALS_FLAG="--no_normals"; shift 1;;

    -h) print_help; exit 0;;
    *) echo -e "${RED}Unknown option: $1${NC}"; print_help; exit 1;;
  esac
done

echo -e "${GREEN}=== SHS-Net Normals Viewer ===${NC}\n"

# Auto-detect runner MODE (wsl/normal)
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

# Validate VIEW_MODE
if [[ "$VIEW_MODE" != "pth" && "$VIEW_MODE" != "scene" && "$VIEW_MODE" != "scene_nn" ]]; then
  echo -e "${RED}Error: --mode must be one of: pth | scene | scene_nn${NC}"
  exit 1
fi

# Validate PC_PTH requirement
if [[ ("$VIEW_MODE" == "pth" || "$VIEW_MODE" == "scene_nn") && -z "$PC_PTH" ]]; then
  echo -e "${RED}Error: --pc_pth is required when --mode is 'pth' or 'scene_nn'.${NC}"
  echo -e "${YELLOW}Example:${NC} $0 --scan ${SCAN_ID} --mode ${VIEW_MODE} --pc_pth /path/to/${SCAN_ID}.pth"
  exit 1
fi

echo -e "${GREEN}✓ Using Python:${NC} ${BLUE}${PYTHON_BIN}${NC}"
echo -e "${GREEN}✓ Using script:${NC} ${BLUE}${PYTHON_SCRIPT}${NC}"
echo -e "${GREEN}✓ scan_id:${NC} ${YELLOW}${SCAN_ID}${NC}"
echo -e "${GREEN}✓ split:${NC} ${YELLOW}${SPLIT}${NC}"
echo -e "${GREEN}✓ cfg:${NC} ${BLUE}${CFG}${NC}"
echo -e "${GREEN}✓ viewer mode:${NC} ${YELLOW}${VIEW_MODE}${NC}"
if [[ -n "$PC_PTH" ]]; then
  echo -e "${GREEN}✓ pc_pth:${NC} ${BLUE}${PC_PTH}${NC}"
fi
echo -e "${GREEN}✓ normals_dir:${NC} ${BLUE}${NORMALS_DIR}${NC}"
if [[ -n "$NORMALS_PATH" ]]; then
  echo -e "${GREEN}✓ normals_path:${NC} ${BLUE}${NORMALS_PATH}${NC}"
fi
echo -e "${GREEN}✓ voxel:${NC} ${YELLOW}${VOXEL}${NC}  ${GREEN}point_size:${NC} ${YELLOW}${POINT_SIZE}${NC}\n"

echo -e "${YELLOW}Close the Open3D window (Q / ESC) to return to terminal.${NC}\n"

ARGS=( --cfg "$CFG" --split "$SPLIT"
       --scan_id "$SCAN_ID"
       --normals_dir "$NORMALS_DIR"
       --mode "$VIEW_MODE"
       --voxel "$VOXEL" --point_size "$POINT_SIZE" 
       --normal_colors
       )

# Only pass pc_pth when provided (and required for pth/scene_nn)
if [[ -n "$PC_PTH" ]]; then
  ARGS+=( --pc_pth "$PC_PTH" )
fi

if [[ -n "$NORMALS_PATH" ]]; then
  ARGS+=( --normals_path "$NORMALS_PATH" )
fi
if [[ -n "$NO_NORMALS_FLAG" ]]; then
  ARGS+=( "$NO_NORMALS_FLAG" )
fi

if [[ "$MODE_CHOICE" == "wsl" ]]; then
  XDG_SESSION_TYPE=x11 "$PYTHON_BIN" "$PYTHON_SCRIPT" "${ARGS[@]}"
else
  "$PYTHON_BIN" "$PYTHON_SCRIPT" "${ARGS[@]}"
fi

echo -e "${GREEN}Python script completed successfully.${NC}"
