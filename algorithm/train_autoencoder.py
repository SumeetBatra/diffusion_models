import torch
import numpy as np
import os
import pickle
import glob
import torch.nn.functional as F

from distutils.util import strtobool
from pathlib import Path
from torch.optim import Adam
from dataset.shaped_elites_dataset import ShapedEliteDataset
from dataset.tensor_elites_dataset import ElitesDataset, postprocess_model
from torch.utils.data import DataLoader
from autoencoders.policy.resnet3d import ResNet3DAutoEncoder
from autoencoders.policy.transformer import TransformerPolicyAutoencoder
from autoencoders.policy.hypernet import HypernetAutoEncoder, ModelEncoder
from RL.actor_critic import Actor
from envs.brax_custom.brax_env import make_vec_env_brax
from attrdict import AttrDict
from utils.brax_utils import compare_rec_to_gt_policy, shared_params
from utils.utilities import log, config_wandb
from functools import partial
from losses.contperceptual import LPIPS


import scipy.stats as stats
import wandb
from datetime import datetime
import argparse
import random
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', choices=['walker2d', 'halfcheetah'])
    parser.add_argument('--model_checkpoint', type=str, default='checkpoints')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--emb_channels', type=int, default=4)
    parser.add_argument('--z_channels', type=int, default=4)
    parser.add_argument('--z_height', type=int, default=4)
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_project', type=str, default='policy_diffusion')
    parser.add_argument('--wandb_run_name', type=str, default='vae_run')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_tag', type=str, default='halfcheetah')
    parser.add_argument('--track_agent_quality', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--merge_obsnorm', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--inp_coef', type=float, default=1)
    parser.add_argument('--kl_coef', type=float, default=1e-6)
    parser.add_argument('--perceptual_loss_coef', type=float, default=0)
    parser.add_argument('--conditional', type=lambda x: bool(strtobool(x)), default=False)

    args = parser.parse_args()
    return args


def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def evaluate_agent_quality(env_cfg: dict,
                           vec_env,
                           gt_params_batch: dict[str, torch.Tensor],
                           rec_policies: list[Actor],
                           obs_norms: dict[str, torch.Tensor],
                           test_batch_size: int,
                           inp_coef: float,
                           device: str,
                           normalize_obs: bool = False):
    avg_measure_mse = 0
    avg_t_test = 0
    avg_orig_reward = 0
    avg_reconstructed_reward = 0
    avg_std_orig_measure = 0
    avg_std_rec_measure = 0
    avg_js_div = 0

    obs_dim = vec_env.single_observation_space.shape[0]
    action_shape = vec_env.single_action_space.shape

    for k in range(test_batch_size):
        gt_agent = Actor(obs_dim, action_shape, normalize_obs=normalize_obs)
        rec_agent = Actor(obs_dim, action_shape, normalize_obs=normalize_obs)
        rec_agent.actor_mean = rec_policies[k]

        actor_weights = {key: gt_params_batch[key][k] for key in gt_params_batch.keys() if 'actor' in key}
        recon_actor_weights = rec_agent.state_dict()

        if normalize_obs:
            # TODO: should remove obs_norms since we are using global obs norm now
            # TODO: or keep it if gloabl obs norm doesn't work for humanoid
            norm_dict = {'obs_normalizer.' + key: obs_norms[key][k] for key in obs_norms.keys()}
            actor_weights.update(norm_dict)
            recon_actor_weights.update(norm_dict)

        # TODO: we should get rid of this if we don't need it anymore
        # TODO: This lets keep for a few core commits
        actor_weights['actor_mean.0.weight'] *= (1 / inp_coef)
        actor_weights['actor_mean.0.bias'] *= (1 / inp_coef)
        recon_actor_weights['actor_mean.actor_mean.0.weight'] *= (1 / inp_coef)
        recon_actor_weights['actor_mean.actor_mean.0.bias'] *= (1 / inp_coef)

        gt_agent.load_state_dict(actor_weights)
        rec_agent.load_state_dict(recon_actor_weights)

        gt_agent.to(device)
        rec_agent.to(device)

        info = compare_rec_to_gt_policy(gt_agent, rec_agent, env_cfg, vec_env, device, deterministic=True)

        avg_measure_mse += info['measure_mse']
        avg_t_test += info['t_test'].pvalue
        avg_orig_reward += info['Rewards/original']
        avg_reconstructed_reward += info['Rewards/reconstructed']
        avg_js_div += info['js_div']
        avg_std_orig_measure += info['Measures/original_std']
        avg_std_rec_measure += info['Measures/reconstructed_std']

    avg_measure_mse /= test_batch_size
    avg_t_test /= test_batch_size
    avg_orig_reward /= test_batch_size
    avg_reconstructed_reward /= test_batch_size
    avg_js_div /= test_batch_size
    avg_std_orig_measure /= test_batch_size
    avg_std_rec_measure /= test_batch_size

    reward_ratio = avg_reconstructed_reward / avg_orig_reward

    log.debug(f'Measure MSE: {avg_measure_mse}')
    log.debug(f'Reward ratio: {reward_ratio}')
    log.debug(f'js_div: {avg_js_div}')

    final_info = {
                    'Behavior/measure_mse_0': avg_measure_mse[0],
                    'Behavior/measure_mse_1': avg_measure_mse[1],
                    'Behavior/orig_reward': avg_orig_reward,
                    'Behavior/rec_reward': avg_reconstructed_reward,
                    'Behavior/reward_ratio': reward_ratio,
                    'Behavior/p-value_0': avg_t_test[0],
                    'Behavior/p-value_1': avg_t_test[1],
                    'Behavior/js_div': avg_js_div,
                    'Behavior/std_orig_measure_0': avg_std_orig_measure[0],
                    'Behavior/std_orig_measure_1': avg_std_orig_measure[1],
                    'Behavior/std_rec_measure_0': avg_std_rec_measure[0],
                    'Behavior/std_rec_measure_1': avg_std_rec_measure[1],
                }
    return final_info


def dataset_factory():
    archive_data_path = 'data/walker2d'
    archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100*.pkl')
    for path in archive_df_paths:
        log.debug(f'loading {path=}')
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


def shaped_elites_dataset_factory(env_name, merge_obsnorm = True, batch_size=32, is_eval=False, inp_coef=0.25):
    archive_data_path = f'data/{env_name}'
    archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            archive_df = pickle.load(f)
            archive_dfs.append(archive_df)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    obs_dim, action_shape = shared_params[env_name]['obs_dim'], np.array([shared_params[env_name]['action_dim']])

    s_elite_dataset = ShapedEliteDataset(archive_dfs,
                                         obs_dim=obs_dim,
                                         action_shape=action_shape,
                                         device=device,
                                         normalize_obs=merge_obsnorm,
                                         is_eval=is_eval,
                                         inp_coef=inp_coef,
                                         eval_batch_size=batch_size if is_eval else None,
                                         )

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


def agent_to_weights_dict(agents: list[Actor]):
    '''Converts a batch of agents of type 'Actor' to a dict of batched weights'''
    weights_dict = {}
    for agent in agents:
        for name, param in agent.named_parameters():
            if name not in weights_dict:
                weights_dict[name] = []
            weights_dict[name].append(param)

    # convert lists to torch tensors
    for key in weights_dict.keys():
        weights_dict[key] = torch.stack(weights_dict[key])

    return weights_dict


def train_autoencoder():
    args = parse_args()

    # experiment name
    exp_name = args.env_name + '_autoencoder_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.conditional:
        exp_name = 'conditional_' + exp_name

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    # get env specific params
    obs_dim, action_shape = shared_params[args.env_name]['obs_dim'], np.array([shared_params[args.env_name]['action_dim']])

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
    model = HypernetAutoEncoder(emb_channels=args.emb_channels,
                                z_channels=args.z_channels,
                                obs_shape=obs_dim,
                                action_shape=action_shape,
                                z_height=args.z_height,
                                conditional=args.conditional,
                                )
    if model_checkpoint is not None:
        print(f'Loading model from checkpoint')
        model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)

    if args.perceptual_loss_coef > 0:
        obs_shape, action_shape = 18, np.array([6])
        encoder_pretrained = ModelEncoder(obs_shape=obs_shape,
                                          action_shape=action_shape,
                                          emb_channels=args.emb_channels,
                                          z_channels=args.z_channels,
                                          z_height=args.z_height,
                                          regress_to_measure=True)
        regressor_path = f'checkpoints/regressor_{args.env_name}.pt'
        log.debug(f'Perceptual loss enabled. Using the classifier stored at {regressor_path}')
        encoder_pretrained.load_state_dict(torch.load(regressor_path))
        encoder_pretrained.to(device)
        # freeze the encoder
        for param in encoder_pretrained.parameters():
            param.requires_grad = False
        # 'perceptual loss' using deep features
        percept_loss = LPIPS(behavior_predictor=encoder_pretrained, spatial=False)
            

    optimizer = Adam(model.parameters(), lr=1e-4)

    mse_loss_func = mse_loss_from_weights_dict

    model_checkpoint_folder = Path(args.model_checkpoint)
    model_checkpoint_folder.mkdir(exist_ok=True)

    train_batch_size, test_batch_size = 32, 8
    dataloader = shaped_elites_dataset_factory(args.env_name, args.merge_obsnorm, batch_size=train_batch_size, \
                                               is_eval=False, inp_coef=args.inp_coef)
    test_dataloader = shaped_elites_dataset_factory(args.env_name, args.merge_obsnorm, batch_size=test_batch_size, \
                                                is_eval=True,  inp_coef=args.inp_coef)
    inp_coef = dataloader.dataset.inp_coef

    if args.track_agent_quality:
        env_cfg = AttrDict({
            'env_name': args.env_name,
            'env_batch_size': 100,
            'num_dims': 2,
            'envs_per_model': 1,
            'seed': 0,
            'num_envs': 100,
        })

        env = make_vec_env_brax(env_cfg)

    epochs = args.num_epochs
    global_step = 0
    for epoch in range(epochs):

        if args.track_agent_quality and epoch % 10 == 0:
            # get a ground truth policy and evaluate it. Then get the reconstructed policy and compare its
            # performance and behavior to the ground truth
            gt_params, gt_measure, obsnorms = next(iter(test_dataloader))
            gt_measure = gt_measure.to(device).to(torch.float32)

            if args.conditional:
                rec_policies, _ = model(gt_params, gt_measure)
            else:
                rec_policies, _ = model(gt_params)

            info = evaluate_agent_quality(env_cfg, env, gt_params, rec_policies, obsnorms, test_batch_size, inp_coef, device, normalize_obs=not args.merge_obsnorm)

            
            # now try to sample a policy with just measures
            if args.conditional:
                rec_policies, _ = model(None, gt_measure)

                info2 = evaluate_agent_quality(env_cfg, env, gt_params, rec_policies, obsnorms, test_batch_size, inp_coef, device, normalize_obs=not args.merge_obsnorm) 

                for key, val in info2.items():
                    info['Conditional_' + key] = val
            # log items to tensorboard and wandb
            if args.use_wandb:
                for key, val in info.items():
                    writer.add_scalar(key, val, global_step + 1)

                info.update({
                    'global_step': global_step + 1,
                    'epoch': epoch + 1
                })

                wandb.log(info)

        epoch_mse_loss = 0
        epoch_kl_loss = 0
        epoch_perceptual_loss = 0
        loss_infos = []
        for step, (policies, measures, _) in enumerate(dataloader):
            optimizer.zero_grad()

            # policies = policies.to(device)
            measures = measures.to(device).to(torch.float32)

            if args.conditional:
                rec_policies, posterior = model(policies, measures)
            else:
                rec_policies, posterior = model(policies)

            policy_mse_loss, loss_info = mse_loss_func(policies, rec_policies)
            kl_loss = posterior.kl().mean()
            loss = policy_mse_loss + args.kl_coef * kl_loss

            if args.perceptual_loss_coef > 0:
                rec_weights_dict = agent_to_weights_dict(rec_policies)
                rec_weights_dict['actor_logstd'] = policies['actor_logstd']
                perceptual_loss = percept_loss(policies, rec_weights_dict).mean()

                epoch_perceptual_loss += perceptual_loss.item()
                loss += args.perceptual_loss_coef * perceptual_loss

            loss.backward()
            # if step % 100 == 0:
            #     print(f'Loss: {loss.item()}')
                # print(f'grad norm: {grad_norm(model)}') TODO: fix this
            optimizer.step()
            global_step += 1

            loss_info['scaled_actor_mean.0.weight'] = (1/((inp_coef)**2))*loss_info['actor_mean.0.weight']
            loss_info['scaled_actor_mean.0.bias'] = (1/((inp_coef)**2))*loss_info['actor_mean.0.bias']
            epoch_mse_loss += policy_mse_loss.item()
            epoch_kl_loss += kl_loss.item()
            loss_infos.append(loss_info)
    
        print(f'Epoch {epoch} MSE Loss: {epoch_mse_loss / len(dataloader)}')
        if args.use_wandb:
            avg_loss_infos = {key: sum([loss_info[key] for loss_info in loss_infos]) / len(loss_infos) for key in loss_infos[0].keys()}

            writer.add_scalar("Loss/mse_loss", epoch_mse_loss / len(dataloader), global_step+1)
            writer.add_scalar("Loss/kl_loss", epoch_kl_loss / len(dataloader), global_step+1)
            writer.add_scalar("Loss/perceptual_loss", epoch_perceptual_loss / len(dataloader), global_step+1)
            wandb.log({
                # TODO: why are we dividing by the length of the dataloader?
                'Loss/mse_loss': epoch_mse_loss / len(dataloader),
                'Loss/kl_loss': epoch_kl_loss / len(dataloader),
                'Loss/perceptual_loss': epoch_perceptual_loss / len(dataloader),
                'epoch': epoch + 1,
                'global_step': global_step + 1
            })
            for key in avg_loss_infos.keys():
                writer.add_scalar(f"Loss/{key}", avg_loss_infos[key], global_step+1)
                wandb.log({f"Loss/{key}": avg_loss_infos[key], "global_step": global_step+1})
            

    print('Saving final model checkpoint...')
    if args.conditional:
        model_name = f'{exp_name}.pt'
    else:
        model_name = f'{exp_name}.pt'
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), model_name))


if __name__ == '__main__':
    train_autoencoder()

# python -m algorithm.train_autoencoder --seed 111