import torch
import numpy as np
import os

from distutils.util import strtobool
from pathlib import Path
from torch.optim import Adam

import torch.nn.functional as F

from utils.utilities import log, config_wandb
# from algorithm.train_autoencoder import shaped_elites_dataset_factory
from autoencoders.policy.hypernet import ModelEncoder
from utils.brax_utils import shared_params


import wandb
from datetime import datetime
import argparse
import random
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from dataset.shaped_elites_dataset import ShapedEliteDataset, WeightNormalizer, LangShapedEliteDataset
import pickle
import pandas
import glob
from utils.archive_utils import archive_df_to_archive, evaluate
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def shaped_elites_dataset_factory(env_name,
                                  batch_size=32,
                                  is_eval=False,
                                  center_data: bool = False,
                                  weight_normalizer: Optional[WeightNormalizer] = None,
                                  use_language: bool = False,
                                  results_folder = "results",
                                  N=-1,
                                  cut_out = False,):
    archive_data_path = f'data/{env_name}'
    archive_dfs = []
    test_archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            log.info(f'Loading archive at {path}')
            archive_df = pickle.load(f)


            if N != -1:
                archive_df = archive_df.sample(N)

            train_df, test_df = train_test_split(archive_df, test_size=0.2, random_state=42)

            archive_dfs.append(train_df)
            test_archive_dfs.append(test_df)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    obs_dim, action_shape = shared_params[env_name]['obs_dim'], np.array([shared_params[env_name]['action_dim']])


    s_elite_dataset = ShapedEliteDataset(archive_dfs, obs_dim=obs_dim,
                                             action_shape=action_shape,
                                             device=device, is_eval=is_eval,
                                             eval_batch_size=batch_size if
                                             is_eval else None,
                                             center_data=center_data,
                                             weight_normalizer=weight_normalizer,
                                             cut_out=cut_out)
    
    s_elite_dataset_test = ShapedEliteDataset(test_archive_dfs, obs_dim=obs_dim,
                                            action_shape=action_shape,
                                             device=device, is_eval=is_eval,
                                             eval_batch_size=batch_size if
                                             is_eval else None,
                                             center_data=center_data,
                                             weight_normalizer=weight_normalizer,
                                             cut_out=cut_out)
    

    weight_normalizer = s_elite_dataset.normalizer
    return DataLoader(s_elite_dataset, batch_size=batch_size, shuffle=not is_eval), DataLoader(s_elite_dataset_test, batch_size=batch_size, shuffle=not is_eval), archive_dfs, weight_normalizer



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
    parser.add_argument('--output_dir', type=str, default='results')

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

    optimizer = Adam(model.parameters(), lr=1e-4)

    model_checkpoint_folder = Path(args.model_checkpoint)
    model_checkpoint_folder.mkdir(exist_ok=True)

    train_batch_size = 32
    # dataloader, _ = shaped_elites_dataset_factory(args.env_name,
    #                                            # args.merge_obsnorm,
    #                                            batch_size=train_batch_size,
    #                                            is_eval=False,
    #                                            inp_coef=args.inp_coef)

    weight_normalizer = None
    dataloader, test_dataloader, train_archive, weight_normalizer = shaped_elites_dataset_factory(args.env_name,
                                                    batch_size=train_batch_size,
                                                    is_eval=False,
                                                    center_data=True,
                                                    cut_out=False,
                                                    results_folder=args.output_dir,
                                                    weight_normalizer=weight_normalizer)


    epochs = args.num_epochs
    global_step = 0
    for epoch in range(epochs):
    
        losses = []
        train_measures = []
        model.train()
        for step, (policies, measures) in enumerate(dataloader):
            optimizer.zero_grad()

            # policies = policies.to(device)
            measures = measures.to(device).to(torch.float32)

            train_measures.append(measures)

            pred_measure, _ = model(policies)


            loss = F.mse_loss(pred_measure, measures)

            loss.backward()

            optimizer.step()
            global_step += 1

            losses.append(loss)
    
        test_losses = []
        test_measures = []
        model.eval()
        with torch.no_grad():
            for step, (policies, measures) in enumerate(test_dataloader):

                # policies = policies.to(device)
                measures = measures.to(device).to(torch.float32)

                test_measures.append(measures)

                pred_measure, _ = model(policies)


                loss = F.mse_loss(pred_measure, measures)


                optimizer.step()
                global_step += 1

                test_losses.append(loss)


        print(f'Epoch {epoch} MSE Loss: {(sum(losses) / len(dataloader)).item()}, Test MSE Loss: {(sum(test_losses) / len(test_dataloader)).item()}')
        print(f'Avg train measure: {torch.mean(torch.cat(train_measures))}, Avg test measure: {torch.mean(torch.cat(test_measures))} \n')
        if args.use_wandb:

            writer.add_scalar("Loss/mse_loss", (sum(losses) / len(dataloader)).item(), global_step+1)
            wandb.log({'Loss/mse_loss': (sum(losses) / len(dataloader)).item(), "global_step": global_step+1, 'epoch': epoch + 1, "Loss/test_mse_loss": (sum(test_losses) / len(test_dataloader)).item()})
            
            

    # print('Saving final model checkpoint...')
    # torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), f'{exp_name}.pt'))


if __name__ == '__main__':
    train_regressor()

# python -m algorithm.train_measure_regressor --seed 111