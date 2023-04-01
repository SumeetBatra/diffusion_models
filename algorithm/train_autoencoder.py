import torch
import numpy as np
import os
import pickle
import glob
import torch.nn.functional as F

from distutils.util import strtobool
from pathlib import Path
from torch.optim import Adam
from dataset.policy_dataset import ElitesDataset, postprocess_model, ShapedEliteDataset
from torch.utils.data import DataLoader
from autoencoders.policy.resnet3d import ResNet3DAutoEncoder
from autoencoders.policy.transformer import TransformerPolicyAutoencoder
from autoencoders.policy.hypernet import HypernetAutoEncoder
from RL.actor_critic import Actor
from envs.brax_custom.brax_env import make_vec_env_brax
from attrdict import AttrDict
from utils.brax_utils import compare_rec_to_gt_policy
from utils.utilities import log, config_wandb


import scipy.stats as stats
import wandb
from datetime import datetime
import argparse
import random
from torch.utils.tensorboard import SummaryWriter
def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def dataset_factory():
    archive_data_path = 'data'
    archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            archive_df = pickle.load(f)
            archive_dfs.append(archive_df)

    scheduler_dfs = []

    scheduler_df_paths = glob.glob(archive_data_path + '/scheduler*100x100*.pkl')
    for path in scheduler_df_paths:
        with open(path, 'rb') as f:
            scheduler_df = pickle.load(f)
            scheduler_dfs.append(scheduler_df)

    mlp_shape = (128, 128, 6)

    dummy_agent = Actor(obs_shape=18, action_shape=np.array([6]))

    elite_dataset = ElitesDataset(archive_dfs, mlp_shape, dummy_agent)

    return DataLoader(elite_dataset, batch_size=32, shuffle=True)


def shaped_elites_dataset_factory(batch_size=32, is_eval=False):
    archive_data_path = 'data'
    archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100_adaptive*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            archive_df = pickle.load(f)
            archive_dfs.append(archive_df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    s_elite_dataset = ShapedEliteDataset(archive_dfs, obs_dim=18, action_shape=np.array([6]), device=device, normalize_obs=True, is_eval=is_eval)

    return DataLoader(s_elite_dataset, batch_size=batch_size, shuffle=not is_eval)


def mse_loss_from_unpadded_params(policy_in_tensors, rec_agents):
    bs = policy_in_tensors.shape[0]

    # convert reconstructed actors to params tensor
    rec_params = np.array([agent.serialize() for agent in rec_agents])
    rec_params = torch.from_numpy(rec_params).reshape(bs, -1)

    mlp_shape = (128, 128, 6)
    dummy_agent = Actor(obs_shape=18, action_shape=np.array([6]))
    # first convert the data from padded -> unpadded params tensors
    params_numpy = np.array([postprocess_model(dummy_agent, tensor, mlp_shape, return_model=False, deterministic=True) for tensor in policy_in_tensors])
    gt_actor_params = torch.from_numpy(params_numpy).reshape(bs, -1)

    return F.mse_loss(gt_actor_params, rec_params)


def mse_loss_from_weights_dict(target_weights_dict: dict, rec_agents: list[Actor]):
    # convert the rec_agents (Actors) into a dict of weights
    pred_weights_dict = {}
    for agent in rec_agents:
        for name, param in agent.named_parameters():
            if name not in pred_weights_dict:
                pred_weights_dict[name] = []
            pred_weights_dict[name].append(param)

    # calculate the loss
    loss = 0
    loss_info = {}
    for key in pred_weights_dict.keys():
        key_loss = F.mse_loss(torch.stack(pred_weights_dict[key]), target_weights_dict[key])
        loss += key_loss
        loss_info[key] = key_loss.item()
    return loss, loss_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints')
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_project', type=str, default='policy_diffusion')
    parser.add_argument('--wandb_run_name', type=str, default='vae_run')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--track_agent_quality', type=lambda x: bool(strtobool(x)), default=True)

    args = parser.parse_args()
    return args


def train_autoencoder():
    # experiment name
    exp_name = 'autoencoder_' + datetime.now().strftime("%Y%m%d-%H%M%S")

    args = parse_args()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.use_wandb:
        writer = SummaryWriter(f"runs/{exp_name}")
        config_wandb(wandb_project=args.wandb_project, wandb_group=args.wandb_group, run_name=args.wandb_run_name, entity=args.wandb_entity)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_checkpoint = None
    model = HypernetAutoEncoder(emb_channels=8, z_channels=4)
    if model_checkpoint is not None:
        print(f'Loading model from checkpoint')
        model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    mse_loss_func = mse_loss_from_weights_dict
    kl_loss_coef = 1e-6

    model_checkpoint_folder = Path(args.model_checkpoint)
    model_checkpoint_folder.mkdir(exist_ok=True)

    dataloader = shaped_elites_dataset_factory(batch_size=32, is_eval=False)
    test_dataloader = shaped_elites_dataset_factory(batch_size=8, is_eval=True)

    if args.track_agent_quality:
        env_cfg = AttrDict({
            'env_name': 'halfcheetah',
            'env_batch_size': 20,
            'num_dims': 2,
            'envs_per_model': 1,
            'seed': 0,
            'num_envs': 20,
        })

        env = make_vec_env_brax(env_cfg)
        obs_dim = env.single_observation_space.shape[0]
        action_shape = env.single_action_space.shape[0]

    epochs = args.num_epochs
    global_step = 0
    for epoch in range(epochs):

        if args.track_agent_quality and epoch % 5 == 0:
            # get a ground truth policy and evaluate it. Then get the reconstructed policy and compare its
            # performance and behavior to the ground truth
            gt_params, gt_measure = next(iter(test_dataloader))
            rec_policies, _ = model(gt_params)

            avg_measure_mse = 0
            avg_t_test = 0
            avg_orig_reward = 0
            avg_reconstructed_reward = 0
            for k in range(8):
                gt_agent = Actor(obs_dim, action_shape, False, False)
                actor_weights = {key: gt_params[key][k] for key in gt_params.keys() if 'actor' in key}
                gt_agent.load_state_dict(actor_weights)
                gt_agent.to(device)

                rec_agent = rec_policies[k]

                info = compare_rec_to_gt_policy(gt_agent, rec_agent, env_cfg, env, device, deterministic=True)

                avg_measure_mse += info['measure_mse']
                avg_t_test += info['t_test'].pvalue
                avg_orig_reward += info['Rewards/original']
                avg_reconstructed_reward += info['Rewards/reconstructed']

            avg_measure_mse /= 8
            avg_t_test /= 8
            avg_orig_reward /= 8
            avg_reconstructed_reward /= 8

            # log.debug(f'T-test p-value: {avg_t_test/8}')
            log.debug(f'Measure MSE: {avg_measure_mse}')

            # log items to tensorboard and wandb
            if args.use_wandb:

                writer.add_scalar('Behaviour/measure_mse_0', avg_measure_mse[0], global_step + 1)
                writer.add_scalar('Behaviour/measure_mse_1', avg_measure_mse[1], global_step + 1)
                writer.add_scalar('Behaviour/orig_reward', avg_orig_reward, global_step + 1)
                writer.add_scalar('Behaviour/rec_reward', avg_reconstructed_reward, global_step + 1)
                writer.add_scalar('Behaviour/p-value_0', avg_t_test[0], global_step + 1)
                writer.add_scalar('Behaviour/p-value_1', avg_t_test[1], global_step + 1)
                wandb.log({
                    'Behaviour/measure_mse_0': avg_measure_mse[0],
                    'Behaviour/measure_mse_1': avg_measure_mse[1],
                    'Behaviour/orig_reward': avg_orig_reward,
                    'Behaviour/rec_reward': avg_reconstructed_reward,
                    'Behaviour/p-value_0': avg_t_test[0],
                    'Behaviour/p-value_1': avg_t_test[1],
                    'global_step': global_step + 1,
                })


        epoch_mse_loss = 0
        epoch_kl_loss = 0        

        for step, (policies, measures) in enumerate(dataloader):
            optimizer.zero_grad()

            # policies = policies.to(device)
            measures = measures.to(device)

            rec_policies, posterior = model(policies)

            policy_mse_loss, loss_info = mse_loss_func(policies, rec_policies)
            kl_loss = posterior.kl().mean()
            loss = policy_mse_loss + kl_loss_coef * kl_loss

            loss.backward()
            # if step % 100 == 0:
            #     print(f'Loss: {loss.item()}')
                # print(f'grad norm: {grad_norm(model)}') TODO: fix this
            optimizer.step()
            global_step += 1

            epoch_mse_loss += policy_mse_loss.item()
            epoch_kl_loss += kl_loss.item()
    
        print(f'Epoch {epoch} MSE Loss: {epoch_mse_loss / len(dataloader)}')
        if args.use_wandb:
            writer.add_scalar("Loss/mse_loss", epoch_mse_loss / len(dataloader), global_step+1)
            writer.add_scalar("Loss/kl_loss", epoch_kl_loss / len(dataloader), global_step+1)
            wandb.log({'Loss/mse_loss': epoch_mse_loss / len(dataloader), "global_step": global_step+1})
            wandb.log({'Loss/kl_loss': epoch_kl_loss / len(dataloader), "global_step": global_step+1})
            for key in loss_info.keys():
                writer.add_scalar(f"Loss/{key}", loss_info[key], global_step+1)
                wandb.log({f"Loss/{key}": loss_info[key], "global_step": global_step+1})

    print('Saving final model checkpoint...')
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), f'{exp_name}_autoencoder.pt'))


if __name__ == '__main__':
    train_autoencoder()

# python -m algorithm.train_autoencoder --seed 111