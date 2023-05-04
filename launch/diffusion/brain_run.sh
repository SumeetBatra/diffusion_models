#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=16:00:00
#SBATCH -N1
#SBATCH --output=tmp2/ldm-%j.log


eval "$(conda shell.bash hook)"
conda activate qd



export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ###########################################################
# centering=True
# env_name=walker
# rn=walker_centering_diffusion
# # ---------------------------------------------------------
# seed=111
# enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-080916_111/model_checkpoints/walker2d_autoencoder_20230502-080916_111.pt

# seed=222
# enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-080913_222/model_checkpoints/walker2d_autoencoder_20230502-080913_222.pt

# seed=333
# enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-080913_333/model_checkpoints/walker2d_autoencoder_20230502-080913_333.pt

# seed=444
# enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-115831_444/model_checkpoints/walker2d_autoencoder_20230502-115831_444.pt
# # ---------------------------------------------------------
# ###########################################################


# ###########################################################
# centering=True
# env_name=humanoid
# rn=humanoid_centering_diffusion
# # ---------------------------------------------------------
# seed=111
# enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-081156_111/model_checkpoints/humanoid_autoencoder_20230502-081156_111.pt

# seed=222
# enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115754_222/model_checkpoints/humanoid_autoencoder_20230502-115754_222.pt

# seed=333
# enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115757_333/model_checkpoints/humanoid_autoencoder_20230502-115757_333.pt

# seed=444
# enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115757_444/model_checkpoints/humanoid_autoencoder_20230502-115757_444.pt
# # ---------------------------------------------------------
# ###########################################################


# ###########################################################
# centering=False
# env_name=walker
# rn=walker_no_centering_diffusion
# # ---------------------------------------------------------
# seed=111
# enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-115838_111/model_checkpoints/walker2d_autoencoder_20230502-115838_111.pt

# seed=222
# enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-185726_222/model_checkpoints/walker2d_autoencoder_20230502-185726_222.pt

# seed=333
# enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-194214_333/model_checkpoints/walker2d_autoencoder_20230502-194214_333.pt

# seed=444
# enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-194611_444/model_checkpoints/walker2d_autoencoder_20230502-194611_444.pt
# # ---------------------------------------------------------
# ###########################################################


###########################################################
centering=False
env_name=humanoid
rn=humanoid_no_centering_diffusion
# ---------------------------------------------------------
# seed=111
# enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115800_111/model_checkpoints/humanoid_autoencoder_20230502-115800_111.pt

# seed=222
# enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115800_222/model_checkpoints/humanoid_autoencoder_20230502-115800_222.pt

# seed=333
# enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115757_333/model_checkpoints/humanoid_autoencoder_20230502-115757_333.pt

seed=444
enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-194214_444/model_checkpoints/humanoid_autoencoder_20230502-194214_444.pt
# ---------------------------------------------------------
###########################################################









srun -c12 python -m algorithm.train --env_name $env_name --use_wandb True --wandb_tag final_diffusion --wandb_group final_diffusion --seed $seed --wandb_run_name $rn --output_dir paper_results --num_epochs 500 --autoencoder_cp_path $enc --centering $centering