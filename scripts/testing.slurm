#!/bin/bash
#SBATCH -J msr3d_exp
#SBATCH --partition=a6000
#SBATCH --qos=long-a6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3        # Use 3 of 4 A6000 GPUs
#SBATCH --mem=256000        # 256GB RAM
#SBATCH --time=3-00:00:00   # 5 days
#SBATCH -o testing.out
#SBATCH -e testing.err

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
export PYTHONPATH="$PWD:$PWD/../Pointcept_main:$PYTHONPATH"
python ./../load_one_item.py --cfg configs/ptv3.yaml --ptv3_cfg ../Pointcept_main/configs/scannet/semseg-pt-v3m1-1-ppt-extreme.py --ptv3_ckpt /lustreFS/data/vcg/pdemetriou/Msqa_Thesis_2025/msr3d/modules/third_party/PTv3/model_best.pth
