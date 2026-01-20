#!/bin/bash

# =========================
#  WSL / X11 Configuration
# =========================

# Only apply these if we detect WSL
if [[ -n "${WSL_DISTRO_NAME-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
  echo "[info] Detected WSL environment."
  # X11 backend variables
  export GDK_BACKEND=x11
  export QT_QPA_PLATFORM=xcb
  export XDG_SESSION_TYPE=x11

  # If DISPLAY is not set, fall back to :0 (WSLg on Windows 11)
  if [[ -z "${DISPLAY-}" ]]; then
    export DISPLAY=:0
  fi

  # Uncomment this if you are using an external X server (Windows 10 + VcXsrv)
  # export DISPLAY=$(grep -oP '(?<=nameserver ).*' /etc/resolv.conf):0
  # export LIBGL_ALWAYS_INDIRECT=1
fi

# =========================
#  Config & Defaults
# =========================

#PYTHON_SCRIPT="situation_visualization_w_instances.py"
PYTHON_SCRIPT="user_interface.py"
PYTHON_BIN="/home/panagiotis/miniconda3/envs/pointcept-torch2.5.0-cu12.4/bin/python3"
MODE_CHOICE=""   # "wsl" or "normal" (auto-detect if empty)

# Colors
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
  -h               Show this help

Examples:
  $0
  $0 -s tools/situation_visualization_w_instances.py
  $0 -p /home/panagiotis/miniconda3/envs/pointcept-torch2.5.0-cu12.4/bin/python \\
     -s /home/panagiotis/msqa/Msqa_Thesis_2025/msr3d/tools/situation_visualization_w_instances.py
EOF
}

# =========================
#  Parse arguments
# =========================

while getopts ":s:p:m:h" opt; do
  case "$opt" in
    s) PYTHON_SCRIPT="$OPTARG" ;;
    p) PYTHON_BIN="$OPTARG" ;;
    m) MODE_CHOICE="$OPTARG" ;;
    h) print_help; exit 0 ;;
    \?) echo -e "${RED}Unknown option: -$OPTARG${NC}" >&2; print_help; exit 1 ;;
    :) echo -e "${RED}Option -$OPTARG requires an argument.${NC}" >&2; exit 1 ;;
  esac
done

echo -e "${GREEN}=== Local Visualization Script ===${NC}\n"

# -------- Auto-detect MODE --------
if [[ -z "${MODE_CHOICE}" ]]; then
  MODE_CHOICE="normal"
  if [[ -n "${WSL_DISTRO_NAME-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
    MODE_CHOICE="wsl"
    echo -e "${YELLOW}[auto-detect] Running in WSL mode${NC}"
  else
    echo -e "${YELLOW}[auto-detect] Running in normal mode${NC}"
  fi
fi

# -------- Check Python script exists --------
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo -e "${RED}Error: Python script '$PYTHON_SCRIPT' not found!${NC}"
  exit 1
fi

echo -e "${GREEN}✓ Using Python:${NC} ${BLUE}${PYTHON_BIN}${NC}"
echo -e "${GREEN}✓ Using script:${NC} ${BLUE}${PYTHON_SCRIPT}${NC}"
echo -e "${GREEN}✓ Mode:${NC} ${YELLOW}${MODE_CHOICE}${NC}\n"

echo -e "${YELLOW}Press 'Q' or 'ESC' to close the Open3D window when it appears.${NC}\n"

# =========================
#  Run Python script
# =========================

if [[ "$MODE_CHOICE" == "wsl" ]]; then
  # Force XDG_SESSION_TYPE=x11 in case the environment overrides it
  XDG_SESSION_TYPE=x11 "$PYTHON_BIN" "$PYTHON_SCRIPT"
else
  "$PYTHON_BIN" "$PYTHON_SCRIPT"
fi

SCRIPT_EXIT_CODE=$?

if [[ $SCRIPT_EXIT_CODE -ne 0 ]]; then
  echo -e "${RED}Python script exited with code ${SCRIPT_EXIT_CODE}.${NC}"
else
  echo -e "${GREEN}Python script completed successfully.${NC}"
fi

exit $SCRIPT_EXIT_CODE
