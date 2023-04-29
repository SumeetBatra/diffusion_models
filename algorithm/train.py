import torch
import numpy as np
import os
import argparse
import wandb
import glob
import pickle
import matplotlib.pyplot as plt
import json
import random

from attrdict import AttrDict
from distutils.util import strtobool
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.utilities import config_wandb, save_cfg
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models.unet import num_to_groups, Unet
from models.cond_unet import ConditionalUNet
from autoencoders.policy.hypernet import HypernetAutoEncoder as AutoEncoder
from algorithm.train_autoencoder import evaluate_agent_quality, shaped_elites_dataset_factory
from dataset.shaped_elites_dataset import ShapedEliteDataset
from diffusion.gaussian_diffusion import GaussianDiffusion, cosine_beta_schedule, linear_beta_schedule
from diffusion.latent_diffusion import LatentDiffusion
from diffusion.ddim import DDIMSampler
from envs.brax_custom.brax_env import make_vec_env_brax
from utils.brax_utils import compare_rec_to_gt_policy, shared_params
from utils.analysis import evaluate_ldm_subsample
from utils.tensor_dict import TensorDict
from utils.utilities import log
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--env_name', choices=['walker2d', 'halfcheetah', 'humanoid', 'ant'])
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results')

    # VAE / LDM hyperparams
    parser.add_argument('--emb_channels', type=int, default=4)
    parser.add_argument('--z_channels', type=int, default=4)
    parser.add_argument('--z_height', type=int, default=4)
    parser.add_argument('--ghn_hid', type=int, default=8)
    parser.add_argument('--enc_fc_hid', type=int, default=64)
    parser.add_argument('--obsnorm_hid', type=int, default=64)
    # wandb args
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_project', type=str, default='policy_diffusion')
    parser.add_argument('--wandb_run_name', type=str, default='diffusion_run')
    parser.add_argument('--wandb_group', type=str, default='debug')
    parser.add_argument('--wandb_entity', type=str, default='qdrl')
    parser.add_argument('--wandb_tag', type=str, default='halfcheetah')
    parser.add_argument('--track_agent_quality', type=lambda x: bool(strtobool(x)), default=True)

    # training hyperparams
    parser.add_argument('--autoencoder_cp_path', type=str)
    # parser.add_argument('--model_checkpoint', type=str, default=None, help='Load an existing diffusion model for additional training')
    
    # misc
    parser.add_argument('--reevaluate_archive_vae', type=lambda x: bool(strtobool(x)), default=True, help='Evaluate the VAE on the entire archive every 50 epochs')
    parser.add_argument('--center_data', type=lambda x: bool(strtobool(x)), default=True,
                        help='Zero center the policy dataset with unit variance')
    parser.add_argument('--clip_obs_rew', type=lambda x: bool(strtobool(x)), default=False,
                        help='Clip obs and rewards b/w -10 and 10 in brax. Set to true if the PPGA archive trained with clipping enabled')
    parser.add_argument('--grad_clip', type=lambda x: bool(strtobool(x)), default=False,
                        help = 'Clip gradients during backprop')

    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg



def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def estimate_component_wise_variance(batch):
    mu = batch.sum() / np.prod(batch.shape)
    var = (batch - mu).square().sum() / np.prod(batch.shape)
    return var


def train(cfg):
    exp_name = cfg.env_name + '_diffusion_model_' + datetime.now().strftime("%Y%m%d-%H%M%S")    
    
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)

    # add experiment name to args
    cfg.exp_name = exp_name
    exp_dir = os.path.join(cfg.output_dir, cfg.env_name)
    os.makedirs(exp_dir, exist_ok=True)

    dm_dir = os.path.join(exp_dir, 'diffusion_model')
    os.makedirs(dm_dir, exist_ok=True)

    dm_dir = os.path.join(dm_dir, exp_name)
    os.makedirs(dm_dir, exist_ok=True)

    cfg.model_checkpoint_folder = os.path.join(dm_dir, 'model_checkpoints')
    os.makedirs(cfg.model_checkpoint_folder, exist_ok=True)

    cfg.image_path = os.path.join(dm_dir, 'images')
    os.makedirs(cfg.image_path, exist_ok=True)

    # set seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    # get env specific params
    obs_dim, action_shape = shared_params[cfg.env_name]['obs_dim'], np.array([shared_params[cfg.env_name]['action_dim']])


    autoencoder_checkpoint_path = Path(cfg.autoencoder_cp_path)
    log.debug(f'Loading autoencoder checkpoint {cfg.autoencoder_cp_path}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 32

    timesteps = 600
    betas = cosine_beta_schedule(timesteps)

    autoencoder = None
    latent_channels = cfg.emb_channels
    latent_size = cfg.z_height

    logvar = torch.full(fill_value=0., size=(timesteps,))
    model = ConditionalUNet(
        in_channels=latent_channels,
        out_channels=latent_channels,
        channels=64,
        n_res_blocks=1,
        attention_levels=[],
        channel_multipliers=[1, 2, 4],
        n_heads=4,
        d_cond=256,
        logvar=logvar
    )
    autoencoder = AutoEncoder(emb_channels=cfg.emb_channels,
                                z_channels=cfg.z_channels,
                                obs_shape=obs_dim,
                                action_shape=action_shape,
                                z_height=cfg.z_height,
                                ghn_hid=cfg.ghn_hid,
                                enc_fc_hid = cfg.enc_fc_hid,
                                obsnorm_hid=cfg.obsnorm_hid,
                                )
    vae_state_dict = torch.load(str(autoencoder_checkpoint_path))
    autoencoder.load_state_dict(vae_state_dict)
    autoencoder.to(device)
    autoencoder.eval()

    gauss_diff = LatentDiffusion(betas, num_timesteps=timesteps, device=device)

    # if model_checkpoint is not None:
    #     print(f'Loading diffusion model from checkpoint...')
    #     model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    train_batch_size, test_batch_size = 32, 50
    train_dataloader, train_archive, weight_normalizer = shaped_elites_dataset_factory(cfg.env_name,
                                                                                       batch_size=train_batch_size,
                                                                                       is_eval=False,
                                                                                       center_data=cfg.center_data)
    test_dataloader, *_ = shaped_elites_dataset_factory(cfg.env_name,
                                                                      batch_size=test_batch_size,
                                                                      is_eval=True,
                                                                      center_data=cfg.center_data,
                                                                      weight_normalizer=weight_normalizer)
    dataset_kwargs = {
        'center_data': cfg.center_data,
        'weight_normalizer': weight_normalizer
    }

    print("Calculating scale factor...")
    gt_params_batch, measures = next(iter(train_dataloader))
    with torch.no_grad():
        batch = autoencoder.encode(gt_params_batch).sample().detach()
        # rescale the embeddings to be unit variance
        std = batch.flatten().std()
        scale_factor = 1. / std
        cfg.scale_factor = scale_factor.item()

    # add args to exp_dir
    with open(os.path.join(dm_dir, 'args.json'), 'w') as f:
        json.dump(cfg, f, indent=4)

    if cfg.use_wandb:
        writer = SummaryWriter(f"runs/{exp_name}")
        config_wandb(wandb_project=cfg.wandb_project,
                     wandb_group=cfg.wandb_group,
                     run_name=cfg.wandb_run_name,
                     entity=cfg.wandb_entity,
                     tags=cfg.wandb_tag,
                     cfg=cfg)

    rollouts_per_agent = 10  # to align ourselves with baselines
    if cfg.track_agent_quality:
        env_cfg = AttrDict({
            'env_name': cfg.env_name,
            'env_batch_size': test_batch_size * rollouts_per_agent,
            'num_dims': shared_params[cfg.env_name]['env_cfg']['num_dims'],
            'seed': 0,
            'clip_obs_rew': cfg.clip_obs_rew
        })

        env = make_vec_env_brax(env_cfg)

        sampler = DDIMSampler(gauss_diff, n_steps=100)

    epochs = cfg.num_epochs
    scale_factor = 1.0
    global_step = 0
    for epoch in range(epochs + 1):

        if cfg.track_agent_quality and epoch % 5 == 0:
            with torch.no_grad():
                # get latents from the LDM using the DDIM sampler. Then use the VAE decoder
                # to get the policies and evaluate their quality
                gt_params_batch, measures = next(iter(test_dataloader))  # get realistic measures to condition on. TODO maybe make this set of measures fixed?
                measures = measures.type(torch.float32).to(device)

                samples = sampler.sample(model, shape=[test_batch_size, latent_channels, latent_size, latent_size], cond=measures)
                samples *= (1 / scale_factor)
                (rec_policies, rec_obsnorms) = autoencoder.decode(samples)

                info = evaluate_agent_quality(env_cfg,
                                            env,
                                            copy.deepcopy(gt_params_batch),
                                            rec_policies,
                                            rec_obsnorms,
                                            test_batch_size,
                                            device=device,
                                            normalize_obs=True,
                                            **dataset_kwargs)

                if epoch % 10 == 0 and cfg.reevaluate_archive_vae:
                    # evaluate the model on the entire archive
                    print('Evaluating model on entire archive...')
                    subsample_results, image_results = evaluate_ldm_subsample(env_name=cfg.env_name, 
                                                                            archive_df=train_archive[0], 
                                                                            ldm=model, 
                                                                            autoencoder=autoencoder, 
                                                                            N=-1, 
                                                                            image_path = cfg.image_path, 
                                                                            suffix = str(epoch), 
                                                                            ignore_first=True, 
                                                                            sampler=sampler, 
                                                                            scale_factor=scale_factor,
                                                                            normalize_obs=True,
                                                                            clip_obs_rew=cfg.clip_obs_rew,
                                                                            uniform_sampling = False,
                                                                            **dataset_kwargs)
                    uniform_subsample_results, uniform_image_results = evaluate_ldm_subsample(env_name=cfg.env_name, 
                                                                            archive_df=train_archive[0], 
                                                                            ldm=model, 
                                                                            autoencoder=autoencoder, 
                                                                            N=-1, 
                                                                            image_path = cfg.image_path, 
                                                                            suffix = "uniform_"+str(epoch), 
                                                                            ignore_first=True, 
                                                                            sampler=sampler, 
                                                                            scale_factor=scale_factor,
                                                                            normalize_obs=True,
                                                                            clip_obs_rew=cfg.clip_obs_rew,
                                                                            uniform_sampling = True,
                                                                            latent_shape = (cfg.z_channels, cfg.z_height, cfg.z_height),
                                                                            **dataset_kwargs)                    
                    for key, val in subsample_results['Reconstructed'].items():
                        info['Archive/' + key] = val
                    for key, val in uniform_subsample_results['Reconstructed'].items():
                        info['Archive/Uniform_' + key] = val

                # log items to tensorboard and wandb
                if cfg.use_wandb:
                    for key, val in info.items():
                        writer.add_scalar(key, val, global_step + 1)

                    info.update({
                        'global_step': global_step + 1,
                        'epoch': epoch + 1
                    })

                    wandb.log(info)
                    if cfg.reevaluate_archive_vae:
                        wandb.log({'Archive/recon_image': wandb.Image(image_results['Reconstructed'], caption=f"Epoch {epoch + 1}")})
                        wandb.log({'Archive/Uniform_recon_image': wandb.Image(uniform_image_results['Reconstructed'], caption=f"Epoch {epoch + 1}")})
        epoch_simple_loss = 0
        epoch_vlb_loss = 0
        epoch_grad_norm = 0
        for step, (policies, measures) in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch_size = measures.shape[0]

            measures = measures.type(torch.float32).to(device)

            with torch.no_grad():
                batch = autoencoder.encode(policies).sample().detach()
                batch *= cfg.scale_factor

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            losses, loss_dict, info_dict = gauss_diff.compute_training_losses(model, batch, t, model_kwargs={'cond': measures})
            loss = losses.mean()

            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            global_step += 1

            epoch_grad_norm += grad_norm(model)
            epoch_simple_loss += loss_dict['losses/simple_loss']
            epoch_vlb_loss += loss_dict['losses/vlb_loss']

            # logging at the per-step timescale
            if cfg.use_wandb:
                wandb.log(info_dict)
                wandb.log({
                    'data/batch_mean': batch.mean().item(),
                    'data/batch_var': batch.var().item(),
                    'global_step': global_step + 1
                })

        # logging at the per-epoch timescale
        log.debug(f'Epoch: {epoch} Simple loss: {epoch_simple_loss / len(train_dataloader)}, Vlb Loss: {epoch_vlb_loss / len(train_dataloader)}')
        if cfg.use_wandb:
            wandb.log({
                'losses/simple_loss': epoch_simple_loss / len(train_dataloader),
                'losses/vlb_loss': epoch_vlb_loss / len(train_dataloader),
                'grad_norm': epoch_grad_norm / len(train_dataloader),
                'epoch': epoch + 1,
                'global_step': global_step + 1
            })

    print('Saving final model checkpoint...')
    cp_name = f'diffusion_model_{cfg.env_name}_{datetime.now().strftime("%Y%m%d-%H%M")}.pt'
    torch.save(model.state_dict(), os.path.join(str(cfg.model_checkpoint_folder), cp_name))
    
    
    # evaluate the final model on the entire archive
    print('Evaluating final model on entire archive...')
    subsample_results, image_results = evaluate_ldm_subsample(env_name=cfg.env_name, 
                                                            archive_df=train_archive[0], 
                                                            ldm=model, 
                                                            autoencoder=autoencoder, 
                                                            N=-1, 
                                                            image_path = cfg.image_path, 
                                                            suffix = "final", 
                                                            ignore_first=False, 
                                                            sampler=sampler, 
                                                            scale_factor=scale_factor,
                                                            normalize_obs=True,
                                                            clip_obs_rew=cfg.clip_obs_rew,
                                                            **dataset_kwargs)
    log.debug(f"Final Reconstruction Results: {subsample_results['Reconstructed']}")
    log.debug(f"Original Archive Reevaluated Results: {subsample_results['Original']}")

    if cfg.use_wandb:
        wandb.log({'Archive/recon_image_final': wandb.Image(image_results['Reconstructed'], caption=f"Final")})
        wandb.log({'Archive/original_image': wandb.Image(image_results['Original'], caption=f"Final")})
        
        wandb.log({'Archive/' + key : val for key, val in subsample_results['Original'].items()})


if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)