#!/bin/bash
#SBATCH -J estimate_normals
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2        # Use 3 of 4 A6000 GPUs
#SBATCH --mem=32000        # 256GB RAM
#SBATCH --time=20:00:00   # 5 days
#SBATCH -o normals_est.out
#SBATCH -e normals_est.err

module purge

# Load required modules
module load CUDA/12.4.0
#module load Python/3.11.5

# Activate your environment
source /lustreFS/data/vcg/pdemetriou/miniconda3/etc/profile.d/conda.sh
conda activate shsnet

export PYTHONNOUSERSITE=1

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1

# Run your training script
cd /lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/
python SHS-Net/run.py --mode test --gpu 0 --data_set Scannet --ckpt_dirs 001 --ckpt_iters 800
