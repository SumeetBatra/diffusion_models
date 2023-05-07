#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=16:00:00
#SBATCH -N1
#SBATCH --output=tmp2/dm-%j.log


eval "$(conda shell.bash hook)"
conda activate qd



export XLA_PYTHON_CLIENT_PREALLOCATE=false



# ###########################################################
# ghn_hid=8
# center=True
# env=ant
# rn=ant_centering
# # ---------------------------------------------------------
# # seed=111

# # seed=222

# # seed=333

# # seed=444
# # ---------------------------------------------------------
# ###########################################################

# ###########################################################
# ghn_hid=8
# center=False
# env=ant
# rn=ant_no_centering
# # ---------------------------------------------------------
# # seed=111

# # seed=222

# # seed=333

# # seed=444
# # ---------------------------------------------------------
# ###########################################################

# ###########################################################
# ghn_hid=8
# center=True
# env=halfcheetah
# rn=halfcheetah_centering
# # ---------------------------------------------------------
# # seed=111

# # seed=222

# # seed=333

# # seed=444
# # ---------------------------------------------------------
# ###########################################################

# ###########################################################
# ghn_hid=8
# center=False
# env=halfcheetah
# rn=halfcheetah_no_centering
# # ---------------------------------------------------------
# # seed=111

# # seed=222

# # seed=333

# # seed=444
# # ---------------------------------------------------------
# ###########################################################




srun -c12 python -m algorithm.train_autoencoder --env_name $env --use_wandb True --wandb_tag final --wandb_group final --seed $seed --wandb_run_name $rn --output_dir paper_results --num_epochs 1000 --center_data $center --ghn_hid $ghn_hid