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
from dataset.shaped_elites_dataset import ShapedEliteDataset, WeightNormalizer, LangShapedEliteDataset
from torch.utils.data import DataLoader
from autoencoders.policy.hypernet import HypernetAutoEncoder, ModelEncoder
from RL.actor_critic import Actor
from envs.brax_custom.brax_env import make_vec_env_brax
from attrdict import AttrDict
from utils.brax_utils import shared_params, rollout_many_agents, calculate_statistics
from utils.utilities import log, config_wandb
from utils.archive_utils import archive_df_to_archive, evaluate
from losses.contperceptual import LPIPS
from utils.analysis import evaluate_vae_subsample
from utils.tensor_dict import TensorDict, cat_tensordicts


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
    parser.add_argument('--ghn_hid', type=int, default=8)
    parser.add_argument('--enc_fc_hid', type=int, default=64)
    parser.add_argument('--obsnorm_hid', type=int, default=64)

    # wandb
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_project', type=str, default='policy_diffusion')
    parser.add_argument('--wandb_run_name', type=str, default='vae_run')
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_tag', type=str, default='halfcheetah')
    parser.add_argument('--track_agent_quality', type=lambda x: bool(strtobool(x)), default=True)
    # loss function hyperparams
    parser.add_argument('--kl_coef', type=float, default=1e-6)
    parser.add_argument('--use_perceptual_loss', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--perceptual_loss_coef', type=float, default=1e-4)
    parser.add_argument('--regressor_path', type=str, default=None, help='Path to regressor model checkpoint required for perceptual loss')
    parser.add_argument('--conditional', type=lambda x: bool(strtobool(x)), default=False)
    # misc
    parser.add_argument('--reevaluate_archive_vae', type=lambda x: bool(strtobool(x)), default=True, help='Evaluate the VAE on the entire archive every 50 epochs')
    parser.add_argument('--load_from_checkpoint', type=str, default=None, help='Load an existing model from a checkpoint for additional training')
    parser.add_argument('--center_data', type=lambda x: bool(strtobool(x)), default=True, help='Zero center the policy dataset with unit variance')
    parser.add_argument('--clip_obs_rew', type=lambda x: bool(strtobool(x)), default=True, help='Clip obs and rewards b/w -10 and 10 in brax. Set to true if the PPGA archive trained with clipping enabled')
    parser.add_argument('--cut_out', type=lambda x: bool(strtobool(x)), default=False, help='cut out elites from the archive that have measure between [0.5,0.5] and [0.6,0.6]')
    parser.add_argument('--average_elites', type=lambda x: bool(strtobool(x)), default=False, help='Average the elites in the archive to get a single elite for each measure, advised to do it for cut_out=True')
    parser.add_argument('--grad_clip', type=lambda x: bool(strtobool(x)), default=True,
                        help = 'Clip gradients during backprop')

    args = parser.parse_args()
    return args


def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def evaluate_agent_quality(env_cfg: dict,
                           vec_env,
                           gt_params_batch: TensorDict,
                           rec_policies: list[Actor],
                           rec_obs_norms: TensorDict,
                           test_batch_size: int,
                           device: str,
                           normalize_obs: bool = False,
                           center_data: bool = False,
                           weight_normalizer: Optional[WeightNormalizer] = None):

    obs_dim = vec_env.single_observation_space.shape[0]
    action_shape = vec_env.single_action_space.shape

    recon_params_batch = [TensorDict(p.state_dict()) for p in rec_policies]
    recon_params_batch = cat_tensordicts(recon_params_batch)
    recon_params_batch.update(rec_obs_norms)

    if center_data:
        assert weight_normalizer is not None and isinstance(weight_normalizer, WeightNormalizer)
        gt_params_batch = weight_normalizer.denormalize(gt_params_batch)
        recon_params_batch = weight_normalizer.denormalize(recon_params_batch)

    if normalize_obs:
        recon_params_batch['obs_normalizer.obs_rms.var'] = torch.exp(recon_params_batch['obs_normalizer.obs_rms.logstd'] * 2)
        recon_params_batch['obs_normalizer.obs_rms.count'] = gt_params_batch['obs_normalizer.obs_rms.count']
        if 'obs_normalizer.obs_rms.logstd' in gt_params_batch:
            del gt_params_batch['obs_normalizer.obs_rms.logstd']
        if 'obs_normalizer.obs_rms.std' in gt_params_batch:
            del gt_params_batch['obs_normalizer.obs_rms.std']
        if 'obs_normalizer.obs_rms.mean' in gt_params_batch:
            del recon_params_batch['obs_normalizer.obs_rms.logstd']


    recon_params_batch['actor_logstd'] = gt_params_batch['actor_logstd']

    gt_agents = [Actor(obs_dim, action_shape, normalize_obs=normalize_obs).to(device) for _ in range(len(gt_params_batch))]
    rec_agents = [Actor(obs_dim, action_shape, normalize_obs=normalize_obs).to(device) for i in range(len(recon_params_batch))]
    for i in range(len(gt_params_batch)):
        gt_agents[i].load_state_dict(gt_params_batch[i])
        rec_agents[i].load_state_dict(recon_params_batch[i])

    # batch-evaluate the ground-truth agents
    gt_rewards, gt_measures = rollout_many_agents(gt_agents, env_cfg, vec_env, device, normalize_obs=normalize_obs, verbose=False)

    # batch-evaluate the reconstructed agents
    rec_rewards, rec_measures = rollout_many_agents(rec_agents, env_cfg, vec_env, device, normalize_obs=normalize_obs, verbose=False)

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
                    'Behavior/measure_mse_2': avg_measure_mse[2] if env_cfg.num_dims == 3 else 0,
                    'Behavior/orig_reward': avg_orig_reward,
                    'Behavior/rec_reward': avg_reconstructed_reward,
                    'Behavior/reward_ratio': reward_ratio,
                    'Behavior/p-value_0': avg_t_test[0],
                    'Behavior/p-value_1': avg_t_test[1],
                    'Behavior/p-value_2': avg_t_test[2] if env_cfg.num_dims == 3 else 0,
                    'Behavior/js_div': avg_js_div,
                    'Behavior/std_orig_measure_0': avg_std_orig_measure[0],
                    'Behavior/std_orig_measure_1': avg_std_orig_measure[1],
                    'Behavior/std_orig_measure_2': avg_std_orig_measure[2] if env_cfg.num_dims == 3 else 0,
                    'Behavior/std_rec_measure_0': avg_std_rec_measure[0],
                    'Behavior/std_rec_measure_1': avg_std_rec_measure[1],
                    'Behavior/std_rec_measure_2': avg_std_rec_measure[2] if env_cfg.num_dims == 3 else 0,
                }
    return final_info


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

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            log.info(f'Loading archive at {path}')
            archive_df = pickle.load(f)

            if cut_out:
                print('Cutting out the middle of the archive')
                ln_before_cut = len(archive_df)
                # ignore the elites that are in the middle of the archive
                archive_df = archive_df[
                    ~((archive_df['measure_0'] > 0.2) & (archive_df['measure_1'] > 0.2)
                    & (archive_df['measure_0'] < 0.6) & (archive_df['measure_1'] < 0.6))]
                print(f'Cut out {ln_before_cut - len(archive_df)} elites')

            if N != -1:
                archive_df = archive_df.sample(N)


            archive_dfs.append(archive_df)

    if use_language:
        text_label_paths = sorted(glob.glob(archive_data_path + '/text_labels_*.pkl'))
        path = text_label_paths[-1]
        with open(path, 'rb') as f:
            log.info(f'Loading text labels from {path}')
            text_labels = pickle.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    obs_dim, action_shape = shared_params[env_name]['obs_dim'], np.array([shared_params[env_name]['action_dim']])

    if is_eval:
        archive_df = pandas.concat(archive_dfs)
        # compute a lower dimensional cvt archive from the original dataset
        soln_dim = archive_df.filter(regex='solution*').to_numpy().shape[1]
        cells = batch_size
        ranges = [(0.0, 1.0)] * shared_params[env_name]['env_cfg']['num_dims']

        # load centroids if they were previously calculated. Maintains consistency across runs
        centroids = None
        centroids_path = f'{results_folder}/{env_name}/centroids.npy'
        if os.path.exists(centroids_path):
            log.info(f'Existing centroids found at {centroids_path}. Loading centroids...')
            centroids = np.load(centroids_path)
        cvt_archive = archive_df_to_archive(archive_df,
                                            type='cvt',
                                            solution_dim=soln_dim,
                                            cells=cells,
                                            ranges=ranges,
                                            custom_centroids=centroids)
        np.save(centroids_path, cvt_archive.centroids)
        # overload the archive_dfs variable with the new archive_df containing only solutions corresponding to the
        # centroids
        archive_dfs = [cvt_archive.as_pandas(include_solutions=True, include_metadata=True)]

    if use_language:
        s_elite_dataset = LangShapedEliteDataset(archive_dfs, obs_dim=obs_dim,
                                                 action_shape=action_shape,
                                                 device=device,
                                                 is_eval=is_eval,
                                                 eval_batch_size=batch_size if
                                                 is_eval else None,
                                                 center_data=center_data,
                                                 weight_normalizer=weight_normalizer,
                                                 text_labels=text_labels)
    else:
        s_elite_dataset = ShapedEliteDataset(archive_dfs, obs_dim=obs_dim,
                                             action_shape=action_shape,
                                             device=device, is_eval=is_eval,
                                             eval_batch_size=batch_size if
                                             is_eval else None,
                                             center_data=center_data,
                                             weight_normalizer=weight_normalizer,
                                             cut_out=cut_out)

    weight_normalizer = s_elite_dataset.normalizer
    return DataLoader(s_elite_dataset, batch_size=batch_size, shuffle=not is_eval), archive_dfs, weight_normalizer


def mse_loss_from_weights_dict(target_weights_dict: TensorDict, pred_weights_dict: TensorDict):
    loss = 0
    loss_info = {}
    for key in pred_weights_dict.keys():
        key_loss = F.mse_loss(target_weights_dict[key], pred_weights_dict[key])
        loss += key_loss
        loss_info[key] = key_loss.item()

    with torch.no_grad():
        obsnorm_loss = torch.Tensor([loss_info[key] for key in loss_info.keys() if 'obs_normalizer' in key]).sum()
        loss_info.update({
            'mse_loss': (loss - obsnorm_loss).item(),
            'obsnorm_loss': obsnorm_loss.item()
        })
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

    # appending seed to name
    exp_name += '_' + str(args.seed)

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

    weight_normalizer_savepath = os.path.join(exp_dir, 'weight_normalizer.pkl')

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
        config_wandb(wandb_project=args.wandb_project,
                     wandb_group=args.wandb_group,
                     run_name=args.wandb_run_name,
                     entity=args.wandb_entity,
                     tags=args.wandb_tag,
                     cfg=vars(args))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HypernetAutoEncoder(emb_channels=args.emb_channels,
                                z_channels=args.z_channels,
                                obs_shape=obs_dim,
                                action_shape=action_shape,
                                z_height=args.z_height,
                                conditional=args.conditional,
                                ghn_hid=args.ghn_hid,
                                enc_fc_hid = args.enc_fc_hid,
                                obsnorm_hid=args.obsnorm_hid,
                                )

    log.info(f'Total number of parameters in the encoder: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)}')
    log.info(f'Total number of parameters in policy the decoder: {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)}')
    log.info(f'Total number of parameters in obs_norm decoder {sum(p.numel() for p in model.obsnorm_decoder.parameters() if p.requires_grad)}')
    log.info(f'Total number of paramers:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

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

    train_batch_size, test_batch_size = 32, 50

    weight_normalizer = None
    if os.path.exists(weight_normalizer_savepath) and args.center_data:
        log.info(f'Loading existing weight normalizer saved at {weight_normalizer_savepath}')
        weight_normalizer = WeightNormalizer(TensorDict({}), TensorDict({}))
        weight_normalizer.load(weight_normalizer_savepath)

    dataloader, train_archive, weight_normalizer = shaped_elites_dataset_factory(args.env_name,
                                                                                 batch_size=train_batch_size,
                                                                                 is_eval=False,
                                                                                 center_data=args.center_data,
                                                                                 cut_out=args.cut_out,
                                                                                 results_folder=args.output_dir,
                                                                                 weight_normalizer=weight_normalizer)
    test_dataloader, test_archive, *_ = shaped_elites_dataset_factory(args.env_name,
                                                                      batch_size=test_batch_size,
                                                                      is_eval=True,
                                                                      center_data=args.center_data,
                                                                      cut_out=args.cut_out,
                                                                      results_folder=args.output_dir,
                                                                      weight_normalizer=weight_normalizer)

    if not os.path.exists(weight_normalizer_savepath) and args.center_data:
        weight_normalizer.save(weight_normalizer_savepath)

    # log.debug(f'{weight_mean_dict=}, {weight_std_dict=}')
    dataset_kwargs = {
        'center_data': args.center_data,
        'weight_normalizer': weight_normalizer
    }

    rollouts_per_agent = 10  # to align ourselves with baselines
    if args.track_agent_quality:
        env_cfg = AttrDict({
            'env_name': args.env_name,
            'env_batch_size': test_batch_size * rollouts_per_agent,
            'num_dims': shared_params[args.env_name]['env_cfg']['num_dims'],
            'seed': 0,
            'clip_obs_rew': args.clip_obs_rew
        })

        env = make_vec_env_brax(env_cfg)

    epochs = args.num_epochs
    global_step = 0
    for epoch in range(epochs + 1):

        if args.track_agent_quality and epoch % 10 == 0:
            # save the model
            model_name = f'{exp_name}.pt'
            torch.save(model.state_dict(), os.path.join(str(args.model_checkpoint_folder), model_name))


            with torch.no_grad():
                # get a ground truth policy and evaluate it. Then get the reconstructed policy and compare its
                # performance and behavior to the ground truth
                gt_params, gt_measure = next(iter(test_dataloader))
                gt_measure = gt_measure.to(device).to(torch.float32)

                if args.conditional:
                    (rec_policies, rec_obsnorms), _ = model(gt_params, gt_measure)
                else:
                    (rec_policies, rec_obsnorms), _ = model(gt_params)
                rec_obsnorms = TensorDict(rec_obsnorms)

                info = evaluate_agent_quality(env_cfg,
                                              env,
                                              gt_params,
                                              rec_policies,
                                              rec_obsnorms,
                                              test_batch_size,
                                              device=device,
                                              normalize_obs=True,
                                              **dataset_kwargs)

                # now try to sample a policy with just measures
                if args.conditional:
                    (rec_policies, rec_obsnorms), _ = model(None, gt_measure)

                    info2 = evaluate_agent_quality(env_cfg,
                                                   env,
                                                   gt_params,
                                                   rec_policies,
                                                   rec_obsnorms,
                                                   test_batch_size,
                                                   device=device,
                                                   normalize_obs=True,
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
                                                                              normalize_obs=True,
                                                                              average=args.average_elites,
                                                                              clip_obs_rew=args.clip_obs_rew,
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
                    if args.reevaluate_archive_vae and args.env_name != "ant":
                        wandb.log({'Archive/recon_image': wandb.Image(image_results['Reconstructed'], caption=f"Epoch {epoch + 1}")})

        epoch_mse_loss = 0
        epoch_kl_loss = 0
        epoch_norm_mse_loss = 0
        epoch_perceptual_loss = 0
        epoch_grad_norm = 0

        loss_infos = []
        for step, (policies, measures) in enumerate(dataloader):
            optimizer.zero_grad()

            # policies = policies.to(device)
            measures = measures.to(device).to(torch.float32)

            if args.conditional:
                (rec_policies, rec_obsnorms), posterior = model(policies, measures)
            else:
                (rec_policies, rec_obsnorms), posterior = model(policies)

            rec_obsnorms = TensorDict(rec_obsnorms)
            # rec_state_dicts = [TensorDict(policy.state_dict()) for policy in rec_policies]
            rec_state_dicts = {}
            for agent in rec_policies:
                for name, param in agent.named_parameters():
                    if name not in rec_state_dicts:
                        rec_state_dicts[name] = []
                    rec_state_dicts[name].append(param)
            for name, param in rec_state_dicts.items():
                rec_state_dicts[name] = torch.stack(param, dim=0)

            # for i, sd in enumerate(rec_state_dicts):
            #     sd.update(rec_obsnorms[i])
            # batch_rec_state_dict = cat_tensordicts(rec_state_dicts)
            rec_state_dicts.update(rec_obsnorms)
            policy_mse_loss, loss_info = mse_loss_func(policies, rec_state_dicts)

            kl_loss = posterior.kl().mean()

            loss = policy_mse_loss + args.kl_coef * kl_loss

            if args.use_perceptual_loss:
                rec_weights_dict = agent_to_weights_dict(rec_policies)
                rec_weights_dict['actor_logstd'] = policies['actor_logstd']
                perceptual_loss = percept_loss(policies, rec_weights_dict).mean()

                epoch_perceptual_loss += perceptual_loss.item()
                loss += args.perceptual_loss_coef * perceptual_loss

            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            global_step += 1

            epoch_mse_loss += loss_info['mse_loss']
            epoch_kl_loss += kl_loss.item()
            epoch_norm_mse_loss += loss_info['obsnorm_loss']
            loss_infos.append(loss_info)
            epoch_grad_norm += grad_norm(model)


        print(f'Epoch {epoch} MSE Loss: {epoch_mse_loss / len(dataloader)}, ObsNorm MSE Loss: {epoch_norm_mse_loss / len(dataloader)}')
        if args.use_wandb:
            avg_loss_infos = {key: sum([loss_info[key] for loss_info in loss_infos]) / len(loss_infos) for key in loss_infos[0].keys()}

            writer.add_scalar("Loss/mse_loss", epoch_mse_loss / len(dataloader), global_step+1)
            writer.add_scalar("Loss/kl_loss", epoch_kl_loss / len(dataloader), global_step+1)
            writer.add_scalar("Loss/perceptual_loss", epoch_perceptual_loss / len(dataloader), global_step+1)
            writer.add_scalar("Loss/norm_mse_loss", epoch_norm_mse_loss / len(dataloader), global_step+1)
            wandb.log({
                'Loss/mse_loss': epoch_mse_loss / len(dataloader),
                'Loss/kl_loss': epoch_kl_loss / len(dataloader),
                'Loss/perceptual_loss': epoch_perceptual_loss / len(dataloader),
                'Loss/norm_mse_loss': epoch_norm_mse_loss / len(dataloader),
                'grad_norm': epoch_grad_norm / len(dataloader),
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
    subsample_results, image_results = evaluate_vae_subsample(env_name=args.env_name,
                                                                archive_df=train_archive[0],
                                                                model=model,
                                                                N=-1,
                                                                image_path = args.image_path,
                                                                suffix = "final",
                                                                ignore_first=False,
                                                                normalize_obs=True,
                                                                average=args.average_elites,
                                                                clip_obs_rew=args.clip_obs_rew,
                                                                **dataset_kwargs)
    log.debug(f"Final Reconstruction Results: {subsample_results['Reconstructed']}")
    log.debug(f"Original Archive Reevaluated Results: {subsample_results['Original']}")

    if args.use_wandb:
        if args.env_name != "ant":
            wandb.log({'Archive/recon_image_final': wandb.Image(image_results['Reconstructed'], caption=f"Final")})
            wandb.log({'Archive/original_image': wandb.Image(image_results['Original'], caption=f"Final")})
        wandb.log({'Archive/original_' + key : val for key, val in subsample_results['Original'].items()})

if __name__ == '__main__':
    train_autoencoder()

# python -m algorithm.train_autoencoder --seed 111
