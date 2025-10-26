#!/bin/bash
export GDK_BACKEND=x11
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0
# Configurations
HPC_USER="pdemetriou"  # Your HPC username
HPC_HOST="prometheus.cyens.org.cy"  # e.g., hpc.example.edu
HPC_DATA_PATH="/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data"
LOCAL_MOUNT_DIR="$HOME/hpc_msr3d_data"  # Local directory to mount data

# Script paths
PYTHON_SCRIPT="situation_visualization.py"  # Your visualization script name

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== HPC Data Mount and Visualization Script ===${NC}\n"

# Check if sshfs is installed
if ! command -v sshfs &> /dev/null; then
    echo -e "${RED}Error: sshfs is not installed.${NC}"
    echo "Install it with:"
    echo "  macOS: brew install macfuse && brew install gromgit/fuse/sshfs-mac"
    echo "  Linux: sudo apt-get install sshfs  (Ubuntu/Debian)"
    echo "         sudo yum install fuse-sshfs  (CentOS/RHEL)"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Python script '$PYTHON_SCRIPT' not found!${NC}"
    exit 1
fi

# Create mount directory if it doesn't exist
mkdir -p "$LOCAL_MOUNT_DIR"

# Check if already mounted
if mount | grep -q "$LOCAL_MOUNT_DIR"; then
    echo -e "${YELLOW}Data already mounted at $LOCAL_MOUNT_DIR${NC}"
else
    echo -e "${GREEN}Mounting HPC data...${NC}"
    sshfs "${HPC_USER}@${HPC_HOST}:${HPC_DATA_PATH}" "$LOCAL_MOUNT_DIR" \
        -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,follow_symlinks
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully mounted HPC data to $LOCAL_MOUNT_DIR${NC}\n"
    else
        echo -e "${RED}✗ Failed to mount HPC data${NC}"
        exit 1
    fi
fi

# Update paths in Python script temporarily
echo -e "${GREEN}Running visualization script...${NC}\n"

# Create a temporary Python script with updated paths
TEMP_SCRIPT=$(mktemp /tmp/visualize_pcd_XXXXXX.py)
sed "s|/lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/data|${LOCAL_MOUNT_DIR}|g" "$PYTHON_SCRIPT" > "$TEMP_SCRIPT"

# Run the Python script
python3 "$TEMP_SCRIPT"
SCRIPT_EXIT_CODE=$?

# Cleanup
rm -f "$TEMP_SCRIPT"

# Unmount option
echo -e "\n${YELLOW}Data is still mounted at $LOCAL_MOUNT_DIR${NC}"
read -p "Do you want to unmount the data? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Unmounting...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        umount "$LOCAL_MOUNT_DIR"
    else
        fusermount -u "$LOCAL_MOUNT_DIR"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully unmounted${NC}"
    else
        echo -e "${RED}✗ Failed to unmount. Try manually:${NC}"
        echo "  macOS: umount $LOCAL_MOUNT_DIR"
        echo "  Linux: fusermount -u $LOCAL_MOUNT_DIR"
    fi
else
    echo -e "${YELLOW}Data remains mounted. To unmount later, run:${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  umount $LOCAL_MOUNT_DIR"
    else
        echo "  fusermount -u $LOCAL_MOUNT_DIR"
    fi
fi

exit $SCRIPT_EXIT_CODE