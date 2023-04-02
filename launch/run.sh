#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --output=tmp/dm-%j.log

# env setup
seed=111


z_c=8
z_h=16
tags=LDM16

# z_c=4
# z_h=32
# tags=LDM8

# z_c=3
# z_h=64
# tags=LDM4


# train

srun \
python -m algorithm.train_autoencoder \
--seed $seed --use_wandb True --num_epochs 200 \
--merge_obsnorm False --wandb_tag $tags --inp_coef 1 \
--z_channels z_c --z_height z_h
