#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --time=16:00:00
#SBATCH -N1
#SBATCH --output=tmp2/ldm-%j.log


eval "$(conda shell.bash hook)"
conda activate qd



export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export WANDB_DISABLE_SERVICE=true

# NORMAL RUNS
ghn_hid=8
# ###########################################################
# centering=True
# env_name=walker2d
# rn=walker_centering_diffusion
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-080916_111/model_checkpoints/walker2d_autoencoder_20230502-080916_111.pt

# # seed=222
# # enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-080913_222/model_checkpoints/walker2d_autoencoder_20230502-080913_222.pt

# # seed=333
# # enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-080913_333/model_checkpoints/walker2d_autoencoder_20230502-080913_333.pt

# # seed=444
# # enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-115831_444/model_checkpoints/walker2d_autoencoder_20230502-115831_444.pt
# # ---------------------------------------------------------
# ###########################################################


# ###########################################################
# centering=True
# env_name=humanoid
# rn=humanoid_centering_diffusion
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-081156_111/model_checkpoints/humanoid_autoencoder_20230502-081156_111.pt

# # seed=222
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115754_222/model_checkpoints/humanoid_autoencoder_20230502-115754_222.pt

# # seed=333
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115757_333/model_checkpoints/humanoid_autoencoder_20230502-115757_333.pt

# # seed=444
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115757_444/model_checkpoints/humanoid_autoencoder_20230502-115757_444.pt
# # ---------------------------------------------------------
# ###########################################################


# ###########################################################
# centering=False
# env_name=walker2d
# rn=walker_no_centering_diffusion
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-115838_111/model_checkpoints/walker2d_autoencoder_20230502-115838_111.pt

# # seed=222
# # enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-185726_222/model_checkpoints/walker2d_autoencoder_20230502-185726_222.pt

# # seed=333
# # enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-194214_333/model_checkpoints/walker2d_autoencoder_20230502-194214_333.pt

# # seed=444
# # enc=paper_results/walker2d/autoencoder/walker2d_autoencoder_20230502-194611_444/model_checkpoints/walker2d_autoencoder_20230502-194611_444.pt
# # ---------------------------------------------------------
# ###########################################################


# ###########################################################
# centering=False
# env_name=humanoid
# rn=humanoid_no_centering_diffusion
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115800_111/model_checkpoints/humanoid_autoencoder_20230502-115800_111.pt

# # seed=222
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115800_222/model_checkpoints/humanoid_autoencoder_20230502-115800_222.pt

# # seed=333
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-115757_333/model_checkpoints/humanoid_autoencoder_20230502-115757_333.pt

# # seed=444
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230502-194214_444/model_checkpoints/humanoid_autoencoder_20230502-194214_444.pt
# # ---------------------------------------------------------
# ###########################################################


# ###########################################################
# centering=True
# env_name=halfcheetah
# rn=halfcheetah_centering_diffusion
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/halfcheetah/autoencoder/halfcheetah_autoencoder_20230505-181208_111/model_checkpoints/halfcheetah_autoencoder_20230505-181208_111.pt

# # seed=222
# # enc=paper_results/halfcheetah/autoencoder/halfcheetah_autoencoder_20230505-183350_222/model_checkpoints/halfcheetah_autoencoder_20230505-183350_222.pt

# # seed=333
# # enc=paper_results/halfcheetah/autoencoder/halfcheetah_autoencoder_20230505-192409_333/model_checkpoints/halfcheetah_autoencoder_20230505-192409_333.pt

# # seed=444
# # enc=paper_results/halfcheetah/autoencoder/halfcheetah_autoencoder_20230505-195951_444/model_checkpoints/halfcheetah_autoencoder_20230505-195951_444.pt
# # ---------------------------------------------------------
# ###########################################################


# ###########################################################
# centering=True
# env_name=ant
# rn=ant_centering_diffusion
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/ant/autoencoder/ant_autoencoder_20230505-184239_111/model_checkpoints/ant_autoencoder_20230505-184239_111.pt

# # seed=222
# # enc=paper_results/ant/autoencoder/ant_autoencoder_20230505-203035_222/model_checkpoints/ant_autoencoder_20230505-203035_222.pt

# # seed=333
# # enc=paper_results/ant/autoencoder/ant_autoencoder_20230505-205508_333/model_checkpoints/ant_autoencoder_20230505-205508_333.pt

# # seed=444
# # enc=paper_results/ant/autoencoder/ant_autoencoder_20230506-065223_444/model_checkpoints/ant_autoencoder_20230506-065223_444.pt
# # ---------------------------------------------------------
# ###########################################################


# ###########################################################
# centering=False
# env_name=halfcheetah
# rn=halfcheetah_no_centering_diffusion
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/halfcheetah/autoencoder/halfcheetah_autoencoder_20230505-203020_111/model_checkpoints/halfcheetah_autoencoder_20230505-203020_111.pt

# # seed=222
# # enc=paper_results/halfcheetah/autoencoder/halfcheetah_autoencoder_20230506-065220_222/model_checkpoints/halfcheetah_autoencoder_20230506-065220_222.pt

# # seed=333
# # enc=paper_results/halfcheetah/autoencoder/halfcheetah_autoencoder_20230506-072459_333/model_checkpoints/halfcheetah_autoencoder_20230506-072459_333.pt

# # seed=444
# # enc=paper_results/halfcheetah/autoencoder/halfcheetah_autoencoder_20230506-080435_444/model_checkpoints/halfcheetah_autoencoder_20230506-080435_444.pt
# # ---------------------------------------------------------
# ###########################################################


# ###########################################################
# centering=False
# env_name=ant
# rn=ant_no_centering_diffusion
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/ant/autoencoder/ant_autoencoder_20230506-072524_111/model_checkpoints/ant_autoencoder_20230506-072524_111.pt

# # seed=222
# # enc=paper_results/ant/autoencoder/ant_autoencoder_20230506-080435_222/model_checkpoints/ant_autoencoder_20230506-080435_222.pt

# # seed=333
# # enc=paper_results/ant/autoencoder/ant_autoencoder_20230506-092158_333/model_checkpoints/ant_autoencoder_20230506-092158_333.pt

# # seed=444
# # enc=paper_results/ant/autoencoder/ant_autoencoder_20230506-092250_444/model_checkpoints/ant_autoencoder_20230506-092250_444.pt
# # ---------------------------------------------------------
# ###########################################################



# SIZE ABLATIONS
# ###########################################################
# ghn_hid=16
# centering=True
# rn=humanoid_centering_diffusion
# env_name=humanoid
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-035956_111/model_checkpoints/humanoid_autoencoder_20230503-035956_111.pt

# # seed=222
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-035956_222/model_checkpoints/humanoid_autoencoder_20230503-035956_222.pt

# # seed=333
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-035848_333/model_checkpoints/humanoid_autoencoder_20230503-035848_333.pt

# # seed=444
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-040024_444/model_checkpoints/humanoid_autoencoder_20230503-040024_444.pt
# # ---------------------------------------------------------
# ###########################################################



# ###########################################################
# ghn_hid=16
# centering=False
# rn=humanoid_no_centering_diffusion
# env_name=humanoid
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-040024_111/model_checkpoints/humanoid_autoencoder_20230503-040024_111.pt

# # seed=222
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-060020_222/model_checkpoints/humanoid_autoencoder_20230503-060020_222.pt

# # seed=333
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-062624_333/model_checkpoints/humanoid_autoencoder_20230503-062624_333.pt

# # seed=444
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-070557_444/model_checkpoint/humanoid_autoencoder_20230503-070557_444.pt
# # ---------------------------------------------------------
# ###########################################################



# ###########################################################
# ghn_hid=32
# centering=True
# rn=humanoid_centering_diffusion
# env_name=humanoid
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-072924_111/model_checkpoints/humanoid_autoencoder_20230503-072924_111.pt

# # seed=222
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-073802_222/model_checkpoints/humanoid_autoencoder_20230503-073802_222.pt

# # seed=333
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-082033_333/model_checkpoints/humanoid_autoencoder_20230503-082033_333.pt

# # seed=444
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-085756_444/model_checkpoints/humanoid_autoencoder_20230503-085756_444.pt
# # ---------------------------------------------------------
# ###########################################################



# ###########################################################
# ghn_hid=32
# centering=False
# rn=humanoid_no_centering_diffusion
# env_name=humanoid
# # ---------------------------------------------------------
# # seed=111
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-085758_111/model_checkpoints/humanoid_autoencoder_20230503-085758_111.pt

# # seed=222
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-100139_222/model_checkpoints/humanoid_autoencoder_20230503-100139_222.pt

# # seed=333
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-101531_333/model_checkpoints/humanoid_autoencoder_20230503-101531_333.pt

# # seed=444
# # enc=paper_results/humanoid/autoencoder/humanoid_autoencoder_20230503-101531_444/model_checkpoints/humanoid_autoencoder_20230503-101531_444.pt
# # ---------------------------------------------------------
# ###########################################################






srun -c12 python -m algorithm.train --env_name $env_name --use_wandb True --wandb_tag final_diffusion --wandb_group final_diffusion --seed $seed --wandb_run_name $rn --output_dir paper_results --num_epochs 500 --autoencoder_cp_path $enc --center_data $centering --ghn_hid $ghn_hid