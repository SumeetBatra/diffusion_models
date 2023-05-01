#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate qd

export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m algorithm.train --env_name walker2d --use_wandb True --num_epochs 500 --wandb_group ldm_sanity --ghn_hid 64 --enc_fc_hid 256 --seed 456 --wandb_run_name ldm_walker2d --reevaluate_archive_vae False --autoencoder_cp_path data/walker2d_autoencoder_20230428-132823.pt --use_language True --language_model flan-t5-small
