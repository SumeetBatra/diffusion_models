#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --output=tmp/quad-%j.log

# env setup
seed=333
# train

srun \
python -m algorithm.train_autoencoder \
--seed $seed
