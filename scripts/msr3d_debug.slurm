#!/bin/bash
#SBATCH -o msr3d_debug.txt      # output file
#SBATCH -e msr3d_debug.err      # error file
#SBATCH -J msr3d     # job name
#SBATCH --partition=defq   # partition
#SBATCH --qos=normal       # priority queue
#SBATCH --ntasks=1         # number of tasks
#SBATCH --cpus-per-task=4  # CPU cores per task
#SBATCH --gres=gpu:2       # number of GPUs
#SBATCH --mem=128000        # memory in MB
#SBATCH --time=24:00     # time limit (1 day, 12 hours)

module purge

# Load required modules
module load CUDA/12.4.0
#module load Python/3.11.5

# Activate your environment
source /lustreFS/data/vcg/pdemetriou/miniconda3/etc/profile.d/conda.sh
conda activate ptv3

export PYTHONNOUSERSITE=1
export HF_HOME=/lustreFS/data/vcg/pdemetriou/hf_cache
export TRANSFORMERS_CACHE=/lustreFS/data/vcg/pdemetriou/hf_cache
export HF_DATASETS_CACHE=/lustreFS/data/vcg/pdemetriou/hf_cache
export WANDB_API_KEY="69a13645442fec7ee7ba8e1a60fa320a26c3f0f8"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1

# Run your training script
cd /lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d
./msr3d.sh
