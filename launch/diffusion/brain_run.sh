#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=09:00:00
#SBATCH -N1
#SBATCH -c4
#SBATCH --output=tmp/ldm-%j.log

ENV_NAME='walker2d'
SEED=0

RUN_NAME='policy_diffusion_'$ENV_NAME"_seed_"$SEED
echo $RUN_NAME

export XLA_PYTHON_CLIENT_PREALLOCATE=false

srun python -m algorith.train --env_name=$ENV_NAME \
                              --use_wandb=True \
                              --wandb_project=policy_diffusion \
                              --wandb_run_name=$RUN_NAME \
                              --wandb_group=default \
                              --wandb_tag=$ENV_NAME \
                              --num_epochs=200 \
                              --emb_channels=4 \
                              --z_channels=4 \
                              --z_height=4 \
                              --autoencoder_cp_path=./checkpoints/autoencoder_"$ENV_NAME".pt \
                              --model_checkpoint=None \
