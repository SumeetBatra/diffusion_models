#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=07:00:00
#SBATCH -N1
#SBATCH --output=tmp/dm-%j.log


eval "$(conda shell.bash hook)"
conda activate qd


tags=plablmo

pl=0

# pl=0.01

# pl=0.0001



export XLA_PYTHON_CLIENT_PREALLOCATE=false

for seed in 123 456 789; do
    srun -c4 python -m algorithm.train_autoencoder --env_name halfcheetah --use_wandb True --num_epochs 200 --wandb_tag $tags --perceptual_loss $pl  --seed $seed &
done

wait
