#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --time=16:00:00
#SBATCH -N1
#SBATCH --output=tmp/dm-%j.log


eval "$(conda shell.bash hook)"
conda activate qd


# z=4
# z=8
z=16

# kc=1e-4
# kc=1e-8

export XLA_PYTHON_CLIENT_PREALLOCATE=false


srun -c12 python -m algorithm.train_autoencoder --env_name humanoid --use_wandb True --num_epochs 500 --wandb_group test --ghn_hid 64 --seed 456 --wandb_run_name vae_run_hmn_common_enc_heatmap --z_channels $z --z_height $z --emb_channels $z --use_perceptual_loss 0 --merge_obsnorm False --reevaluate_archive_vae True 