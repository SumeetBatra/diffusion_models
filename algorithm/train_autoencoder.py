import torch
import numpy as np
import os
import pickle
import pandas
import glob
import torch.nn.functional as F

from typing import Optional
from distutils.util import strtobool
from torch.optim import Adam
from dataset.shaped_elites_dataset import ShapedEliteDataset
from torch.utils.data import DataLoader
from autoencoders.policy.hypernet import HypernetAutoEncoder, ModelEncoder
from RL.actor_critic import Actor
from envs.brax_custom.brax_env import make_vec_env_brax
from attrdict import AttrDict
from utils.brax_utils import shared_params, rollout_many_agents, calculate_statistics
from utils.utilities import log, config_wandb
from utils.archive_utils import archive_df_to_archive
from losses.contperceptual import LPIPS
from utils.analysis import evaluate_vae_subsample


import scipy.stats as stats
import wandb
from datetime import datetime
import argparse
import json
import random
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--env_name', choices=['walker2d', 'halfcheetah', 'humanoid', 'humanoid_crawl', 'ant'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results')
    # VAE params
    parser.add_argument('--emb_channels', type=int, default=4)
    parser.add_argument('--z_channels', type=int, default=4)
    parser.add_argument('--z_height', type=int, default=4)
    parser.add_argument('--ghn_hid', type=int, default=32)
    
    # wandb
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_project', type=str, default='policy_diffusion')
    parser.add_argument('--wandb_run_name', type=str, default='vae_run')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_tag', type=str, default='halfcheetah')
    parser.add_argument('--track_agent_quality', type=lambda x: bool(strtobool(x)), default=True)
    # loss function hyperparams
    parser.add_argument('--merge_obsnorm', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--inp_coef_w', type=float, default=1)
    parser.add_argument('--inp_coef_b', type=float, default=1)
    parser.add_argument('--kl_coef', type=float, default=1e-6)
    parser.add_argument('--use_perceptual_loss', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--perceptual_loss_coef', type=float, default=1e-4)
    parser.add_argument('--regressor_path', type=str, default=None, help='Path to regressor model checkpoint required for perceptual loss')
    parser.add_argument('--conditional', type=lambda x: bool(strtobool(x)), default=False)
    # misc
    parser.add_argument('--reevaluate_archive_vae', type=lambda x: bool(strtobool(x)), default=True, help='Evaluate the VAE on the entire archive every 50 epochs')
    parser.add_argument('--load_from_checkpoint', type=str, default=None, help='Load an existing model from a checkpoint for additional training')
    parser.add_argument('--center_data', type=lambda x: bool(strtobool(x)), default=True, help='Zero center the policy dataset with unit variance')

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
                           rec_obs_norms: dict[str, torch.Tensor],
                           test_batch_size: int,
                           inp_coefs: tuple[float],
                           device: str,
                           normalize_obs: bool = False,
                           center_data: bool = False,
                           weight_denormalizer = None,
                           weight_normalizer = None,
                           obsnorm_denormalizer = None,
                           obsnorm_normalizer = None,):

    obs_dim = vec_env.single_observation_space.shape[0]
    action_shape = vec_env.single_action_space.shape
    (inp_coef_w, inp_coef_b) = inp_coefs

    # load all relevant data for ground truth and reconstructed agents
    gt_agents, rec_agents = [], []
    for k in range(test_batch_size):
        gt_agent = Actor(obs_dim, action_shape, normalize_obs=normalize_obs).to(device)
        rec_agent = Actor(obs_dim, action_shape, normalize_obs=normalize_obs).to(device)

        actor_weights = {key: gt_params_batch[key][k] for key in gt_params_batch.keys() if 'actor' in key}
        recon_actor_weights = rec_policies[k].state_dict()

        actor_weights['actor_mean.0.weight'] *= (1 / inp_coef_w)
        actor_weights['actor_mean.0.bias'] *= (1 / inp_coef_b)
        recon_actor_weights['actor_mean.0.weight'] *= (1 / inp_coef_w)
        recon_actor_weights['actor_mean.0.bias'] *= (1 / inp_coef_b)

        if center_data:
            weight_denormalizer(actor_weights)
            weight_denormalizer(recon_actor_weights)
            

        if normalize_obs:
            obs_norms_cpy = {'obs_rms.mean' : obs_norms['obs_rms.mean'][k],
                'obs_rms.var' : obs_norms['obs_rms.var'][k],
                'obs_rms.count' : obs_norms['obs_rms.count'][k],}

            rec_obs_norms_cpy = {'obs_rms.mean' : rec_obs_norms['obs_rms.mean'][k],
                            'obs_rms.var' : torch.exp(rec_obs_norms['obs_rms.logstd'][k] * 2),
                            'obs_rms.count' : obs_norms['obs_rms.count'][k],}
            
            if center_data:
                obsnorm_denormalizer(obs_norms_cpy)
                obsnorm_denormalizer(rec_obs_norms_cpy)

            gt_norm_dict = {'obs_normalizer.' + key: obs_norms_cpy[key] for key in obs_norms_cpy.keys()}
            rec_norm_dict = {'obs_normalizer.' + key: rec_obs_norms_cpy[key] for key in rec_obs_norms_cpy.keys()}
            actor_weights.update(gt_norm_dict)
            recon_actor_weights.update(rec_norm_dict)

        recon_actor_weights['actor_logstd'] = actor_weights['actor_logstd']

        gt_agent.load_state_dict(actor_weights)
        rec_agent.load_state_dict(recon_actor_weights)

        gt_agents.append(gt_agent)
        rec_agents.append(rec_agent)

    # batch-evaluate the ground-truth agents
    gt_rewards, gt_measures = rollout_many_agents(gt_agents, env_cfg, vec_env, device, normalize_obs = normalize_obs)

    # batch-evaluate the reconstructed agents
    rec_rewards, rec_measures = rollout_many_agents(rec_agents, env_cfg, vec_env, device, normalize_obs = normalize_obs)

    # calculate statistics based on results
    info = calculate_statistics(gt_rewards, gt_measures, rec_rewards, rec_measures)
    avg_measure_mse = info['measure_mse']
    avg_t_test = info['t_test'].pvalue
    avg_orig_reward = info['Rewards/original']
    avg_reconstructed_reward = info['Rewards/reconstructed']
    avg_js_div = info['js_div']
    avg_std_orig_measure = info['Measures/original_std']
    avg_std_rec_measure = info['Measures/reconstructed_std']

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


def shaped_elites_dataset_factory(env_name,
                                  merge_obsnorm = True,
                                  batch_size=32,
                                  is_eval=False,
                                  inp_coefs: tuple[float] = (1.0, 1.0),
                                  center_data: bool = False,
                                  weight_mean_dict: Optional[dict] = None,
                                  weight_std_dict: Optional[dict] = None,
                                  obsnorm_mean_dict: Optional[dict] = None,
                                  obsnorm_std_dict: Optional[dict] = None,):
    archive_data_path = f'data/{env_name}'
    archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            log.info(f'Loading archive at {path}')
            archive_df = pickle.load(f)
            archive_dfs.append(archive_df)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    obs_dim, action_shape = shared_params[env_name]['obs_dim'], np.array([shared_params[env_name]['action_dim']])

    if is_eval:
        archive_df = pandas.concat(archive_dfs)
        # compute a lower dimensional cvt archive from the original dataset
        soln_dim = archive_df.filter(regex='solution*').to_numpy().shape[1]
        cells = batch_size
        ranges = [(0.0, 1.0)] * shared_params[env_name]['env_cfg']['num_dims']
        cvt_archive = archive_df_to_archive(archive_df,
                                            type='cvt',
                                            solution_dim=soln_dim,
                                            cells=cells,
                                            ranges=ranges)
        # overload the archive_dfs variable with the new archive_df containing only solutions corresponding to the
        # centroids
        archive_dfs = [cvt_archive.as_pandas(include_solutions=True, include_metadata=True)]

    s_elite_dataset = ShapedEliteDataset(archive_dfs,
                                         obs_dim=obs_dim,
                                         action_shape=action_shape,
                                         device=device,
                                         normalize_obs=merge_obsnorm,
                                         is_eval=is_eval,
                                         inp_coefs=inp_coefs,
                                         eval_batch_size=batch_size if is_eval else None,
                                         center_data=center_data,
                                         weight_mean_dict=weight_mean_dict,
                                         weight_std_dict=weight_std_dict,
                                         obsnorm_mean_dict=obsnorm_mean_dict,
                                         obsnorm_std_dict=obsnorm_std_dict)

    weights_mean, weights_std = s_elite_dataset.weight_mean_dict, s_elite_dataset.weight_std_dict
    denormalizer_weight = s_elite_dataset.denormalize_weights
    normalizer_weight = s_elite_dataset.normalize_weights
    obsnorm_mean, obsnorm_std = s_elite_dataset.obs_norm_mean_dict, s_elite_dataset.obs_norm_std_dict
    denormalizer_obsnorm = s_elite_dataset.denormalize_obsnorm
    normalizer_obsnorm = s_elite_dataset.normalize_obsnorm
    return DataLoader(s_elite_dataset, batch_size=batch_size, shuffle=not is_eval), archive_dfs, weights_mean, \
        weights_std, denormalizer_weight, normalizer_weight, obsnorm_mean, obsnorm_std, denormalizer_obsnorm, normalizer_obsnorm


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

def mse_from_norm_dict(target_weights_dict: dict, rec_weight_dict: dict):
    loss = 0
    loss_info = {}
    for key in rec_weight_dict.keys():
        key_loss = F.mse_loss(target_weights_dict[key], rec_weight_dict[key])
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

    # add experiment name to args
    args.exp_name = exp_name
    exp_dir = os.path.join(args.output_dir, args.env_name)
    os.makedirs(exp_dir, exist_ok=True)

    vae_dir = os.path.join(exp_dir, 'autoencoder')
    os.makedirs(vae_dir, exist_ok=True)

    vae_dir = os.path.join(vae_dir, exp_name)
    os.makedirs(vae_dir, exist_ok=True)

    args.model_checkpoint_folder = os.path.join(vae_dir, 'model_checkpoints')
    os.makedirs(args.model_checkpoint_folder, exist_ok=True)

    args.image_path = os.path.join(vae_dir, 'images')
    os.makedirs(args.image_path, exist_ok=True)

    # add args to exp_dir
    with open(os.path.join(vae_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

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
    model = HypernetAutoEncoder(emb_channels=args.emb_channels,
                                z_channels=args.z_channels,
                                obs_shape=obs_dim,
                                action_shape=action_shape,
                                z_height=args.z_height,
                                conditional=args.conditional,
                                ghn_hid=args.ghn_hid,
                                )

    if args.load_from_checkpoint is not None:
        log.info(f'Loading an existing model saved at {args.load_from_checkpoint}')
        model.load_state_dict(torch.load(args.load_from_checkpoint))
    model.to(device)

    obs_shape, action_shape = obs_dim, action_shape = shared_params[args.env_name]['obs_dim'], np.array([shared_params[args.env_name]['action_dim']])

    if args.use_perceptual_loss:
        encoder_pretrained = ModelEncoder(obs_shape=obs_shape,
                                            action_shape=action_shape,
                                            emb_channels=args.emb_channels,
                                            z_channels=args.z_channels,
                                            z_height=args.z_height,
                                            regress_to_measure=True)
        log.debug(f'Perceptual loss enabled. Using the classifier stored at {args.regressor_path}')
        encoder_pretrained.load_state_dict(torch.load(args.regressor_path))
        encoder_pretrained.to(device)
        # freeze the encoder
        for param in encoder_pretrained.parameters():
            param.requires_grad = False
        # 'perceptual loss' using deep features
        percept_loss = LPIPS(behavior_predictor=encoder_pretrained, spatial=False)

    optimizer = Adam(model.parameters(), lr=1e-4)

    mse_loss_func = mse_loss_from_weights_dict
    norm_mse_loss_func = mse_from_norm_dict

    inp_coefs = (args.inp_coef_w, args.inp_coef_b)

    train_batch_size, test_batch_size = 32, 8
    dataloader, train_archive, weight_mean_dict, weight_std_dict, weight_denormalizer, weight_normalizer, obsnorm_mean_dict, obsnorm_std_dict, obsnorm_denormalizer, obsnorm_normalizer = shaped_elites_dataset_factory( \
                                                args.env_name, args.merge_obsnorm, batch_size=train_batch_size, \
                                                is_eval=False, inp_coefs=inp_coefs, center_data=args.center_data)
    test_dataloader, test_archive, _, _, _, _, _, _, _, _ = shaped_elites_dataset_factory(args.env_name, args.merge_obsnorm, batch_size=test_batch_size, \
                                                is_eval=True,  inp_coefs=inp_coefs, center_data=args.center_data, \
                                                    weight_mean_dict=weight_mean_dict, weight_std_dict=weight_std_dict, \
                                                        obsnorm_mean_dict=obsnorm_mean_dict, obsnorm_std_dict=obsnorm_std_dict)

    # log.debug(f'{weight_mean_dict=}, {weight_std_dict=}')
    dataset_kwargs = {
        'inp_coefs': inp_coefs,
        'center_data': args.center_data,
        'weight_denormalizer': weight_denormalizer,
        'weight_normalizer': weight_normalizer,
        'obsnorm_denormalizer': obsnorm_denormalizer,
        'obsnorm_normalizer': obsnorm_normalizer,
    }

    rollouts_per_agent = 10  # to align ourselves with baselines
    if args.track_agent_quality:
        env_cfg = AttrDict({
            'env_name': args.env_name,
            'env_batch_size': test_batch_size * rollouts_per_agent,
            'num_dims': shared_params[args.env_name]['env_cfg']['num_dims'],
            'seed': 0,
        })

        env = make_vec_env_brax(env_cfg)

    epochs = args.num_epochs
    global_step = 0
    for epoch in range(epochs + 1):

        if args.track_agent_quality and epoch % 10 == 0:
            # get a ground truth policy and evaluate it. Then get the reconstructed policy and compare its
            # performance and behavior to the ground truth
            gt_params, gt_measure, obsnorms = next(iter(test_dataloader))
            gt_measure = gt_measure.to(device).to(torch.float32)

            if args.conditional:
                rec_policies, _ = model(gt_params, obsnorms, gt_measure)
            else:
                (rec_policies, rec_obsnorm), _ = model(gt_params, obsnorms)

            info = evaluate_agent_quality(env_cfg, 
                                          env, gt_params, 
                                          rec_policies, 
                                          obsnorms, 
                                          rec_obsnorm, 
                                          test_batch_size, 
                                          device=device, 
                                          normalize_obs=not args.merge_obsnorm, 
                                          **dataset_kwargs)
            
            # now try to sample a policy with just measures
            if args.conditional:
                rec_policies, _ = model(None, gt_measure)

                info2 = evaluate_agent_quality(env_cfg, 
                                               env, 
                                               gt_params, 
                                               rec_policies, 
                                               obsnorms, 
                                               rec_obsnorm,
                                               test_batch_size, 
                                               device=device, 
                                               normalize_obs=not args.merge_obsnorm, 
                                               **dataset_kwargs)

                for key, val in info2.items():
                    info['Conditional_' + key] = val

            if epoch % 50 == 0 and args.reevaluate_archive_vae:
                # evaluate the model on the entire archive
                print('Evaluating model on entire archive...')
                subsample_results, image_results = evaluate_vae_subsample(env_name=args.env_name, 
                                                                          archive_df=train_archive[0], 
                                                                          model=model, 
                                                                          N=-1, 
                                                                          image_path = args.image_path, 
                                                                          suffix = str(epoch), 
                                                                          ignore_first=True,
                                                                          normalize_obs=not args.merge_obsnorm, 
                                                                          **dataset_kwargs)
                for key, val in subsample_results['Reconstructed'].items():
                    info['Archive/' + key] = val
                
            # log items to tensorboard and wandb
            if args.use_wandb:
                for key, val in info.items():
                    writer.add_scalar(key, val, global_step + 1)

                info.update({
                    'global_step': global_step + 1,
                    'epoch': epoch + 1
                })

                wandb.log(info)
                if args.reevaluate_archive_vae:
                    wandb.log({'Archive/recon_image': wandb.Image(image_results['Reconstructed'], caption=f"Epoch {epoch + 1}")})

        epoch_mse_loss = 0
        epoch_kl_loss = 0
        epoch_norm_mse_loss = 0
        epoch_norm_kl_loss = 0
        epoch_perceptual_loss = 0
        loss_infos = []
        for step, (policies, measures, obsnorms) in enumerate(dataloader):
            optimizer.zero_grad()

            # policies = policies.to(device)
            measures = measures.to(device).to(torch.float32)

            if args.conditional:
                (rec_policies, rec_obsnorm), (posterior, obsnorm_posterior) = model(policies, obsnorms, measures)
            else:
                (rec_policies, rec_obsnorm), (posterior, obsnorm_posterior) = model(policies, obsnorms)

            policy_mse_loss, loss_info = mse_loss_func(policies, rec_policies)
            norm_mse_loss, norm_loss_info = norm_mse_loss_func(obsnorms, rec_obsnorm)

            kl_loss = posterior.kl().mean()
            norm_kl_loss = obsnorm_posterior.kl().mean()

            loss = policy_mse_loss + args.kl_coef * kl_loss + norm_mse_loss + args.kl_coef * norm_kl_loss

            if args.use_perceptual_loss:
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

            loss_info['scaled_actor_mean.0.weight'] = (1/((inp_coefs[0])**2))*loss_info['actor_mean.0.weight']
            loss_info['scaled_actor_mean.0.bias'] = (1/((inp_coefs[1])**2))*loss_info['actor_mean.0.bias']
            epoch_mse_loss += policy_mse_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_norm_mse_loss += norm_mse_loss.item()
            epoch_norm_kl_loss += norm_kl_loss.item()
            loss_infos.append(loss_info)


    
        print(f'Epoch {epoch} MSE Loss: {epoch_mse_loss / len(dataloader)}, Norm MSE Loss: {epoch_norm_mse_loss / len(dataloader)}')
        if args.use_wandb:
            avg_loss_infos = {key: sum([loss_info[key] for loss_info in loss_infos]) / len(loss_infos) for key in loss_infos[0].keys()}

            writer.add_scalar("Loss/mse_loss", epoch_mse_loss / len(dataloader), global_step+1)
            writer.add_scalar("Loss/kl_loss", epoch_kl_loss / len(dataloader), global_step+1)
            writer.add_scalar("Loss/perceptual_loss", epoch_perceptual_loss / len(dataloader), global_step+1)
            writer.add_scalar("Loss/norm_mse_loss", epoch_norm_mse_loss / len(dataloader), global_step+1)
            writer.add_scalar("Loss/norm_kl_loss", epoch_norm_kl_loss / len(dataloader), global_step+1)
            wandb.log({
                'Loss/mse_loss': epoch_mse_loss / len(dataloader),
                'Loss/kl_loss': epoch_kl_loss / len(dataloader),
                'Loss/perceptual_loss': epoch_perceptual_loss / len(dataloader),
                'Loss/norm_mse_loss': epoch_norm_mse_loss / len(dataloader),
                'Loss/norm_kl_loss': epoch_norm_kl_loss / len(dataloader),
                'epoch': epoch + 1,
                'global_step': global_step + 1
            })
            for key in avg_loss_infos.keys():
                writer.add_scalar(f"Loss/{key}", avg_loss_infos[key], global_step+1)
                wandb.log({f"Loss/{key}": avg_loss_infos[key], "global_step": global_step+1})
            

    print('Saving final model checkpoint...')

    model_name = f'{exp_name}.pt'
    torch.save(model.state_dict(), os.path.join(str(args.model_checkpoint_folder), model_name))

    # evaluate the final model on the entire archive
    print('Evaluating final model on entire archive...')
    subsample_results, image_results = evaluate_vae_subsample(env_name=args.env_name, archive_df=train_archive[0], model=model, N=-1, image_path = args.image_path, suffix = "final", ignore_first=False)
    log.debug(f"Final Reconstruction Results: {subsample_results['Reconstructed']}")
    log.debug(f"Original Archive Reevaluated Results: {subsample_results['Original']}")

    if args.use_wandb:
        wandb.log({'Archive/recon_image_final': wandb.Image(image_results['Reconstructed'], caption=f"Final")})
        wandb.log({'Archive/original_image': wandb.Image(image_results['Original'], caption=f"Final")})
if __name__ == '__main__':
    train_autoencoder()

# python -m algorithm.train_autoencoder --seed 111