#!/bin/bash
#SBATCH --account=gaurav_1048
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16GB
#SBATCH --time=2:30:00
#SBATCH --output=tmp/dm-%j.log

module purge
module load gcc/11.3.0
module load cuda/11.6.2
module load cudnn/8.4.0.27-11.6
eval "$(conda shell.bash hook)"
conda activate qd

tags=plabl
seed=111

# pl=0

# pl=0.01

# pl=0.05

# pl=0.1

# pl=0.5

pl=1


srun python -m algorithm.train_autoencoder --seed $seed --use_wandb True --num_epochs 200 --merge_obsnorm False --wandb_tag $tags --inp_coef 1 --perceptual_loss $pl

