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


srun -c12 python -m algorithm.train --env_name walker2d --use_wandb True --num_epochs 500 --wandb_group ldm_sanity --ghn_hid 64 --enc_fc_hid 256 --seed 456 --wandb_run_name ldm_walker2d --reevaluate_archive_vae True --autoencoder_cp_path results/walker2d/autoencoder/walker2d_autoencoder_20230428-132823/model_checkpoints/walker2d_autoencoder_20230428-132823.pt