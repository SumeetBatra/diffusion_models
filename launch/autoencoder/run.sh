#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --output=tmp/dm-%j.log

# env setup
seed=111


pl=0

# pl=0.001

# pl=0.005

# pl=0.01

# pl=0.05

# pl=0.1

# pl=0.5

# pl=1


tags=plabl

# train

srun \
python -m algorithm.train_autoencoder \
--seed $seed --use_wandb True --num_epochs 200 \
--merge_obsnorm False --wandb_tag $tags --inp_coef 1 \
--perceptual_loss $pl
