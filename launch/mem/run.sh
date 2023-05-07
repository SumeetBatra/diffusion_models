#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --time=16:00:00
#SBATCH -N1
#SBATCH --output=tmp2/dm-%j.log


eval "$(conda shell.bash hook)"
conda activate qd



export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun -c12 python -m utils.metric_calculator