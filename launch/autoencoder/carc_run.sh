#!/bin/bash
#SBATCH --account=gaurav_1048
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
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


group=sizeabl1

# gh=64

# gh=32

# gh=16

gh=8



export XLA_PYTHON_CLIENT_PREALLOCATE=false

for seed in 123 456 789; do
    srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK python -m algorithm.train_autoencoder --env_name halfcheetah --use_wandb True --num_epochs 200 --wandb_group $group --ghn_hid $pl --seed $seed &
done
wait
