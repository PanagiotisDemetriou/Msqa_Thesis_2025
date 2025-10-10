#!/bin/bash

# Parameters
#SBATCH --account=research
#SBATCH --cpus-per-task=32
#SBATCH --error=/lustreFS/data/vcg/pdemetriou/msr3d/jobs/%j/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=msr3d
#SBATCH --mem=100GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/lustreFS/data/vcg/pdemetriou/msr3d/jobs/%j/%j_0_log.out
#SBATCH --partition=defq
#SBATCH --qos=lv1
#SBATCH --signal=USR2@120
#SBATCH --time=1440
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /lustreFS/data/vcg/pdemetriou/msr3d/jobs/%j/%j_%t_log.out --error /lustreFS/data/vcg/pdemetriou/msr3d/jobs/%j/%j_%t_log.err /lustreFS/data/vcg/pdemetriou/miniconda3/envs/pointcept/bin/python -u -m submitit.core._submit /lustreFS/data/vcg/pdemetriou/msr3d/jobs/%j
