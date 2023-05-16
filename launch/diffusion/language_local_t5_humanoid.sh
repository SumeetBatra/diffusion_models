#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate qd

export XLA_PYTHON_CLIENT_PREALLOCATE=false

SEED=456

python -m algorithm.train --env_name humanoid --use_wandb True --num_epochs 500 --wandb_group ldm_sanity --ghn_hid 32 --enc_fc_hid 64 --seed $SEED --wandb_run_name lang_ldm_humanoid_$SEED --reevaluate_archive_vae False --autoencoder_cp_path data/humanoid/model_checkpoints/humanoid_autoencoder_20230503-082033_333.pt --use_language True --language_model flan-t5-small
