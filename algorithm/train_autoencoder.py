import torch
import numpy as np
import os
import pickle
import glob
import torch.nn.functional as F

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


def shaped_elites_dataset_factory():
    archive_data_path = 'data'
    archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100_no_obs_norm*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            archive_df = pickle.load(f)
            archive_dfs.append(archive_df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    s_elite_dataset = ShapedEliteDataset(archive_dfs, obs_dim=18, action_shape=np.array([6]), device=device)

    return DataLoader(s_elite_dataset, batch_size=32, shuffle=True)


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
    for key in pred_weights_dict.keys():
        loss += F.mse_loss(torch.stack(pred_weights_dict[key]), target_weights_dict[key])

    return loss


def train_autoencoder():
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

    model_checkpoint_folder = Path('./checkpoints')
    model_checkpoint_folder.mkdir(exist_ok=True)

    dataloader = shaped_elites_dataset_factory()

    track_agent_quality = True
    if track_agent_quality:
        env_cfg = AttrDict({
            'env_name': 'halfcheetah',
            'env_batch_size': 100,
            'num_dims': 2,
            'envs_per_model': 1,
            'seed': 0,
            'num_envs': 100,
        })

        env = make_vec_env_brax(env_cfg)
        obs_dim = env.single_observation_space.shape[0]
        action_shape = env.single_action_space.shape[0]

    epochs = 40
    global_step = 0
    for epoch in range(epochs):

        if track_agent_quality and epoch % 5 == 0:
            # get a ground truth policy and evaluate it. Then get the reconstructed policy and compare its
            # performance and behavior to the ground truth
            gt_params, gt_measure = next(iter(dataloader))
            gt_agent = Actor(obs_dim, action_shape, False, False)
            actor_weights = {key: gt_params[key][0] for key in gt_params.keys() if 'actor' in key}
            gt_agent.load_state_dict(actor_weights)
            gt_agent.to(device)

            rec_policies, _ = model(gt_params)
            rec_agent = rec_policies[0]

            compare_rec_to_gt_policy(gt_agent, rec_agent, env_cfg, env, device, deterministic=True)

        print(f'{epoch=}')
        print(f'{global_step=}')
        epoch_mse_loss = 0
        epoch_kl_loss = 0        

        for step, (policies, measures) in enumerate(dataloader):
            optimizer.zero_grad()

            # policies = policies.to(device)
            measures = measures.to(device)

            rec_policies, posterior = model(policies)

            policy_mse_loss = mse_loss_func(policies, rec_policies)
            kl_loss = posterior.kl().mean()
            loss = policy_mse_loss + kl_loss_coef * kl_loss

            loss.backward()
            # if step % 100 == 0:
            #     print(f'Loss: {loss.item()}')
                # print(f'grad norm: {grad_norm(model)}') TODO: fix this
            optimizer.step()
            global_step += step

            epoch_mse_loss += policy_mse_loss.item()
            epoch_kl_loss += kl_loss.item()
    
        print(f'Epoch {epoch} MSE Loss: {epoch_mse_loss / len(dataloader)}')



    print('Saving final model checkpoint...')
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), 'autoencoder.pt'))


if __name__ == '__main__':
    train_autoencoder()

