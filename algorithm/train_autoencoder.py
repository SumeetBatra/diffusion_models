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
import scipy.stats as stats
import wandb
from datetime import datetime
import argparse
import random
from tensorboardX import SummaryWriter

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

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            archive_df = pickle.load(f)
            archive_dfs.append(archive_df)


    scheduler_df_paths = glob.glob(archive_data_path + '/scheduler*100x100*.pkl')

    
    with open(scheduler_df_paths[0], 'rb') as f:
        scheduler = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    s_elite_dataset = ShapedEliteDataset(archive_dfs, obs_dim=18, action_shape=np.array([6]), device=device, scheduler=scheduler)

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
    # pred_weights_dict['rms_mean'] = []
    # pred_weights_dict['rms_var'] = []
    for agent in rec_agents:
        for name, param in agent.named_parameters():
            if name not in pred_weights_dict:
                pred_weights_dict[name] = []
            pred_weights_dict[name].append(param)

        # pred_weights_dict['rms_mean'].append(agent.obs_normalizer.obs_rms.mean)
        # pred_weights_dict['rms_var'].append(agent.obs_normalizer.obs_rms.var)

    # calculate the loss
    loss = 0
    for key in pred_weights_dict.keys():
        loss += F.mse_loss(torch.stack(pred_weights_dict[key]), target_weights_dict[key])

    return loss

def enjoy_brax(agent, env, env_cfg, device, deterministic=True):

    obs_mean, obs_var = agent.obs_normalizer.obs_rms.mean, agent.obs_normalizer.obs_rms.var

    obs = env.reset()
    rollout = [env.unwrapped._state]
    total_reward = 0
    measures = torch.zeros(env_cfg.num_dims).to(device)
    done = False
    while not done:
        with torch.no_grad():
            obs = obs.unsqueeze(dim=0).to(device)
            obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)

            if deterministic:
                act = agent.actor_mean(obs)
            else:
                act, _, _ = agent.get_action(obs)
            act = act.squeeze()
            obs, rew, done, info = env.step(act.cpu())
            measures += info['measures']
            rollout.append(env.unwrapped._state)
            total_reward += rew

    # print(f'{total_reward=}')
    # print(f' Rollout length: {len(rollout)}')
    measures /= len(rollout)
    # print(f'Recorded Measures: {measures.cpu().numpy()}')
    return total_reward.detach().cpu().numpy(), measures.cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)

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

    writer = SummaryWriter(f"runs/{exp_name}")
    wandb.init(
        project='policy_diffusion', 
        name=exp_name,
        # config=vars(args),
        sync_tensorboard=True,
        )


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_checkpoint = None
    model = HypernetAutoEncoder(emb_channels=8, z_channels=4)
    if model_checkpoint is not None:
        print(f'Loading model from checkpoint')
        model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    mse_loss_func = mse_loss_from_weights_dict
    kl_loss_coef = 1e-4

    model_checkpoint_folder = Path(args.model_checkpoint)
    model_checkpoint_folder.mkdir(exist_ok=True)

    dataloader = shaped_elites_dataset_factory()
    env_cfg = AttrDict({
        'env_name': 'halfcheetah',
        'env_batch_size': None,
        'num_dims': 2,
        'envs_per_model': 1,
        'seed': 0,
        'num_envs': 1,
    })

    env = make_vec_env_brax(env_cfg)
    obs_dim = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]

    # disc_start = 50001
    # kl_weight = 1e-6
    # disc_weight = 0.5
    # loss_func = LPIPSWithDiscriminator(disc_start, kl_weight=kl_weight, disc_weight=disc_weight)
    # optimizer2 = Adam(loss_func.discriminator.parameters(), lr=1e-3)


    epochs = args.num_epochs
    global_step = 0
    for epoch in range(epochs):


        if epoch % 10 == 0:
            # get next batch of policies
            eval_params, eval_measure = next(iter(dataloader))
            agent = Actor(obs_dim, action_shape, True, True)
            rec_policies, posterior = model(eval_params)
            avg_p_values = np.zeros((1,2))
            avg_orig_rewards = 0
            avg_rec_rewards = 0
            for i in range(5): 
                actor_weights = {key:eval_params[key][i] for key in eval_params.keys() if 'actor' in key}
                actor_weights['obs_normalizer.obs_rms.mean'] = eval_params['rms_mean'][i]
                actor_weights['obs_normalizer.obs_rms.var'] = eval_params['rms_var'][i]
                actor_weights['obs_normalizer.obs_rms.count'] = eval_params['rms_count'][i]

                actor_weights['return_normalizer.return_rms.mean'] = agent.return_normalizer.return_rms.mean
                actor_weights['return_normalizer.return_rms.var'] = agent.return_normalizer.return_rms.var
                actor_weights['return_normalizer.return_rms.count'] = agent.return_normalizer.return_rms.count

                agent.load_state_dict(actor_weights)
                agent.to(device)
                # print(f'Sampled measure from elite: {eval_measure[i]}')
                # print("Running an elite policy from the dataset...")
                total_rewards = []
                true_eval_measures = []
                for l in range(5):
                    total_reward, true_eval_measure = enjoy_brax(agent, env, env_cfg, device)
                    total_rewards.append(total_reward)
                    true_eval_measures.append(true_eval_measure)

                # print(f'Average reward: {np.mean(total_rewards)}, std: {np.std(total_rewards)}')
                avg_orig_rewards += np.mean(total_rewards)

                agent.actor_mean = rec_policies[i].actor_mean
                # agent = rec_policies[i]
                # print("Running a policy from the autoencoder...")

                recon_total_rewards = []
                recon_true_eval_measures = []
                for l in range(5):
                    total_reward, true_eval_measure = enjoy_brax(agent, env, env_cfg, device)
                    recon_total_rewards.append(total_reward)
                    recon_true_eval_measures.append(true_eval_measure)

                # print(f'Average reward: {np.mean(recon_total_rewards)}, std: {np.std(recon_total_rewards)}')
                avg_rec_rewards += np.mean(recon_total_rewards)

                result = stats.ttest_ind(true_eval_measures, recon_true_eval_measures, equal_var = False)
                avg_p_values += result.pvalue
            print(f'Avg original reward: {avg_orig_rewards/5}')
            print(f'Avg reconstructed reward: {avg_rec_rewards/5}')
            print(f'T-test p-value: {avg_p_values/5}')
            writer.add_scalar('Rewards/original', avg_orig_rewards/5, global_step+1)
            writer.add_scalar('Rewards/reconstructed', avg_rec_rewards/5, global_step+1)
            writer.add_scalar('dist_shift/p-value', avg_p_values/5, global_step+1)
            print("--------------------------------------------------------------------")

        # print(f'{epoch=}')
        # print(f'{global_step=}')
        epoch_mse_loss = 0
        epoch_kl_loss = 0        

        for step, (policies, measures) in enumerate(dataloader):
            optimizer.zero_grad()

            # policies = policies.to(device)
            measures = measures.to(device)

            rec_policies, posterior = model(policies)
            # loss = loss_func(batch, img_out, posterior, global_step, 0)
            # loss += loss_func(batch, img_out, posterior, global_step, 1)
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
        writer.add_scalar("Loss/mse_loss", epoch_mse_loss / len(dataloader), global_step+1)
        writer.add_scalar("Loss/kl_loss", epoch_kl_loss / len(dataloader), global_step+1)



    print('Saving final model checkpoint...')
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), f'{exp_name}_autoencoder.pt'))


if __name__ == '__main__':
    train_autoencoder()

# python -m algorithm.train_autoencoder --seed 111