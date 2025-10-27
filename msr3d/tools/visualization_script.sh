#!/bin/bash

# X11 Display Configuration
export GDK_BACKEND=x11
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0
export XDG_SESSION_TYPE=x11

# Configurations
HPC_USER="pdemetriou"
HPC_HOST="prometheus.cyens.org.cy"
HPC_DATA_PATH="/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data"
LOCAL_MOUNT_DIR="$HOME/hpc_msr3d_data"

# Script paths
PYTHON_SCRIPT="situation_visualization_w_instances.py"
MODE_CHOICE=""

# Colors for output
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
  -m <mode>        Mode: "wsl" or "normal" (auto-detect if omitted)
  -h               Show this help

Before running this script, manually mount the HPC data:
  ${BLUE}sshfs ${HPC_USER}@${HPC_HOST}:${HPC_DATA_PATH} ${LOCAL_MOUNT_DIR} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,follow_symlinks${NC}

To unmount later:
  ${BLUE}fusermount -u ${LOCAL_MOUNT_DIR}${NC}  (Linux/WSL)
  ${BLUE}umount ${LOCAL_MOUNT_DIR}${NC}         (macOS)
EOF
}

# Parse arguments
while getopts ":s:m:h" opt; do
  case "$opt" in
    s) PYTHON_SCRIPT="$OPTARG" ;;
    m) MODE_CHOICE="$OPTARG" ;;
    h) print_help; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; print_help; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

echo -e "${GREEN}=== HPC Visualization Script ===${NC}\n"

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

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script '$PYTHON_SCRIPT' not found!${NC}"
    exit 1
fi

# Check if data is mounted
if ! mount | grep -q "$LOCAL_MOUNT_DIR"; then
    echo -e "${RED}Error: Data not mounted at $LOCAL_MOUNT_DIR${NC}\n"
    echo -e "${YELLOW}Please mount the HPC data first:${NC}"
    echo -e "${BLUE}sshfs ${HPC_USER}@${HPC_HOST}:${HPC_DATA_PATH} ${LOCAL_MOUNT_DIR} -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,follow_symlinks${NC}\n"
    
    # Create mount directory if needed
    if [ ! -d "$LOCAL_MOUNT_DIR" ]; then
        echo -e "${YELLOW}Creating mount directory: $LOCAL_MOUNT_DIR${NC}"
        mkdir -p "$LOCAL_MOUNT_DIR"
    fi
    
    exit 1
fi

echo -e "${GREEN}✓ Data is mounted at $LOCAL_MOUNT_DIR${NC}\n"

# Update paths in Python script temporarily
echo -e "${GREEN}Running visualization script...${NC}"
echo -e "${YELLOW}Mode: $MODE_CHOICE${NC}"
echo -e "${YELLOW}Press 'Q' or 'ESC' to close each visualization window${NC}\n"

# Create a temporary Python script with updated paths
TEMP_SCRIPT=$(mktemp /tmp/visualize_pcd_XXXXXX.py)
sed "s|/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data|${LOCAL_MOUNT_DIR}|g" "$PYTHON_SCRIPT" > "$TEMP_SCRIPT"

# Run the Python script with proper X11 configuration
if [[ "$MODE_CHOICE" == "wsl" ]]; then
    XDG_SESSION_TYPE=x11 python3 "$TEMP_SCRIPT"
else
    python3 "$TEMP_SCRIPT"
fi

SCRIPT_EXIT_CODE=$?

# Cleanup
rm -f "$TEMP_SCRIPT"

echo -e "\n${GREEN}Script completed!${NC}"
echo -e "${YELLOW}To unmount the data, run:${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${BLUE}umount $LOCAL_MOUNT_DIR${NC}"
else
    echo -e "${BLUE}fusermount -u $LOCAL_MOUNT_DIR${NC}"
fi

exit $SCRIPT_EXIT_CODE