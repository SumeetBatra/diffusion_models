import torch
import numpy as np
import os

from distutils.util import strtobool
from pathlib import Path
from torch.optim import Adam

import torch.nn.functional as F

from utils.utilities import log, config_wandb
from algorithm.train_autoencoder import shaped_elites_dataset_factory
from autoencoders.policy.hypernet import ModelEncoder
from utils.brax_utils import shared_params


import wandb
from datetime import datetime
import argparse
import random
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, choices=['halfcheetah', 'walker2d', 'humanoid', 'humanoid_crawl'])
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--emb_channels', type=int, default=4)
    parser.add_argument('--z_channels', type=int, default=4)
    parser.add_argument('--z_height', type=int, default=4)
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_project', type=str, default='policy_diffusion')
    parser.add_argument('--wandb_run_name', type=str, default='measure_regressor_run')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_tag', type=str, default='halfcheetah')
    parser.add_argument('--track_agent_quality', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--merge_obsnorm', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--inp_coef', type=float, default=1)
    parser.add_argument('--kl_coef', type=float, default=1e-6)

    args = parser.parse_args()
    return args


def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def train_regressor():

    args = parse_args()

    # experiment name
    exp_name = args.env_name + '_regressor_' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    # env related params
    obs_shape, action_shape = shared_params[args.env_name]['obs_dim'], np.array([shared_params[args.env_name]['action_dim']])

    if args.use_wandb:
        writer = SummaryWriter(f"runs/{exp_name}")
        config_wandb(wandb_project=args.wandb_project, \
                     wandb_group=args.wandb_group, \
                        run_name=args.wandb_run_name, \
                            entity=args.wandb_entity, \
                                tags=args.wandb_tag, \
                                    cfg = vars(args) \
                                        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_checkpoint = None
    model = ModelEncoder(obs_shape=obs_shape,
                         action_shape=action_shape,
                         emb_channels=args.emb_channels,
                         z_channels=args.z_channels,
                         z_height=args.z_height,
                         regress_to_measure=True)
    if model_checkpoint is not None:
        print(f'Loading model from checkpoint')
        model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    model_checkpoint_folder = Path(args.model_checkpoint)
    model_checkpoint_folder.mkdir(exist_ok=True)

    train_batch_size = 32
    dataloader, _ = shaped_elites_dataset_factory(args.env_name,
                                               args.merge_obsnorm,
                                               batch_size=train_batch_size,
                                               is_eval=False,
                                               inp_coef=args.inp_coef)

    epochs = args.num_epochs
    global_step = 0
    for epoch in range(epochs):
    
        losses = []
        for step, (policies, measures, _) in enumerate(dataloader):
            optimizer.zero_grad()

            # policies = policies.to(device)
            measures = measures.to(device).to(torch.float32)

            pred_measure, _ = model(policies)


            loss = F.mse_loss(pred_measure, measures)

            loss.backward()

            optimizer.step()
            global_step += 1

            losses.append(loss)
    
        print(f'Epoch {epoch} MSE Loss: {(sum(losses) / len(dataloader)).item()}')
        if args.use_wandb:

            writer.add_scalar("Loss/mse_loss", (sum(losses) / len(dataloader)).item(), global_step+1)
            wandb.log({'Loss/mse_loss': (sum(losses) / len(dataloader)).item(), "global_step": global_step+1, 'epoch': epoch + 1})
            
            

    print('Saving final model checkpoint...')
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), f'{exp_name}.pt'))


if __name__ == '__main__':
    train_regressor()

# python -m algorithm.train_measure_regressor --seed 111