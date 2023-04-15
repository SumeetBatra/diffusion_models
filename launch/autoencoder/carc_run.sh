#!/bin/bash
#SBATCH --account=gaurav_1048
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=tmp/dm-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=3

module purge
module load gcc/11.3.0
module load cuda/11.6.2
module load cudnn/8.4.0.27-11.6
eval "$(conda shell.bash hook)"
conda activate qd

su_charge=$(scontrol show job $SLURM_JOB_ID -dd | grep -oP '(?<=billing=).*(?=,gres/gpu=)')
# log the SU charge
echo "estimated max su charge $su_charge"


tags=plabl2

# pl=0

# pl=0.01

# pl=0.05

pl=0.1


export XLA_PYTHON_CLIENT_PREALLOCATE=false

for seed in 123 456 789; do
    srun --cpus-per-task=8 python -m algorithm.train_autoencoder --env_name halfcheetah --use_wandb True --num_epochs 200 --wandb_tag $tags --perceptual_loss $pl  --seed $seed &
done

wait
