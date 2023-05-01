#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12G
#SBATCH --time=16:00:00
#SBATCH -N1
#SBATCH --output=tmp/dm-%j.log


eval "$(conda shell.bash hook)"
conda activate qd



export XLA_PYTHON_CLIENT_PREALLOCATE=false

seed=123

srun -c12 python -m algorithm.train_autoencoder --env_name humanoid --use_wandb True --wandb_tag final --wandb_group final --seed $seed --wandb_run_name humanoid_centering --output_dir paper_results --num_epochs 500