import torch
import numpy as np
import os
import argparse
import wandb
import glob
import pickle
import matplotlib.pyplot as plt

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

from utils.utilities import log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_diffusion', type=lambda x: bool(strtobool(x)), default=True, help='Use latent diffusion or standard (improved) diffusion')
    # wandb args
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_project', type=str, default='policy_diffusion')
    parser.add_argument('--wandb_run_name', type=str, default='diffusion_run')
    parser.add_argument('--wandb_group', type=str, default='debug')
    parser.add_argument('--wandb_entity', type=str, default='qdrl')
    parser.add_argument('--wandb_tag', type=str, default='halfcheetah')
    parser.add_argument('--track_agent_quality', type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument('--merge_obsnorm', type=lambda x: bool(strtobool(x)), default=True)
    # training hyperparams
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inp_coef', type=float, default=1.0)
    # VAE / LDM hyperparams
    parser.add_argument('--emb_channels', type=int, default=4)
    parser.add_argument('--z_channels', type=int, default=4)
    parser.add_argument('--z_height', type=int, default=4)
    parser.add_argument('--autoencoder_cp_path', type=str)
    parser.add_argument('--model_checkpoint', type=str, default=None, help='Load an existing diffusion model for additional training')
    # Task specific args
    parser.add_argument('--env_name', choices=['walker2d', 'halfcheetah'])

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
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    save_and_sample_every = 1000

    model_checkpoint_folder = Path('./checkpoints')
    model_checkpoint_folder.mkdir(exist_ok=True)
    model_checkpoint = cfg.model_checkpoint

    exp_name = 'diffusion_model_' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # get env specific params
    obs_dim, action_shape = shared_params[cfg.env_name]['obs_dim'], np.array([shared_params[cfg.env_name]['action_dim']])


    if cfg.use_wandb:
        writer = SummaryWriter(f"runs/{exp_name}")
        config_wandb(wandb_project=cfg.wandb_project,
                     wandb_group=cfg.wandb_group,
                     run_name=cfg.wandb_run_name,
                     entity=cfg.wandb_entity,
                     tags=cfg.wandb_tag,
                     cfg=cfg)
    autoencoder_checkpoint_path = Path(cfg.autoencoder_cp_path)
    log.debug(f'Loading autoencoder checkpoint {cfg.autoencoder_cp_path}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_size = 32
    channels = 1
    batch_size = 32

    timesteps = 600
    betas = cosine_beta_schedule(timesteps)

    autoencoder = None
    if cfg.latent_diffusion:
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
                                  z_height=cfg.z_height)
        autoencoder.load_state_dict(torch.load(str(autoencoder_checkpoint_path)))
        autoencoder.to(device)
        autoencoder.eval()

        gauss_diff = LatentDiffusion(betas, num_timesteps=timesteps, device=device)
    else:
        model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4),
            out_dim=2 * channels
        )
        gauss_diff = GaussianDiffusion(betas, num_timesteps=timesteps, device=device)

    if model_checkpoint is not None:
        print(f'Loading diffusion model from checkpoint...')
        model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    train_batch_size, test_batch_size = 32, 8
    train_dataloader, _ = shaped_elites_dataset_factory(cfg.env_name, cfg.merge_obsnorm, batch_size=train_batch_size, \
                                               is_eval=False, inp_coef=cfg.inp_coef)
    test_dataloader, _ = shaped_elites_dataset_factory(cfg.env_name, cfg.merge_obsnorm, batch_size=test_batch_size, \
                                                is_eval=True,  inp_coef=cfg.inp_coef)
    inp_coef = train_dataloader.dataset.inp_coef

    if cfg.track_agent_quality:
        env_cfg = AttrDict({
            'env_name': cfg.env_name,
            'env_batch_size': 100,
            'num_dims': 2,
            'envs_per_model': 1,
            'seed': 0,
            'num_envs': 100,
        })

        env = make_vec_env_brax(env_cfg)

        sampler = DDIMSampler(gauss_diff, n_steps=100)

    epochs = cfg.num_epochs
    scale_factor = 1.0
    global_step = 0
    for epoch in range(epochs):

        if cfg.track_agent_quality and epoch % 5 == 0 and epoch != 0:  # skip the 0th epoch b/c we need to calculate the scale factor first
            # get latents from the LDM using the DDIM sampler. Then use the VAE decoder
            # to get the policies and evaluate their quality
            gt_params_batch, measures, obsnorms = next(iter(test_dataloader))  # get realistic measures to condition on. TODO maybe make this set of measures fixed?
            measures = measures.type(torch.float32).to(device)

            samples = sampler.sample(model, shape=[test_batch_size, latent_channels, latent_size, latent_size], cond=measures)
            samples *= (1 / scale_factor)
            rec_policies = autoencoder.decode(samples)

            info = evaluate_agent_quality(env_cfg, env, gt_params_batch, rec_policies, obsnorms, test_batch_size, inp_coef, device, normalize_obs=not cfg.merge_obsnorm)

            # log items to tensorboard and wandb
            if cfg.use_wandb:
                for key, val in info.items():
                    writer.add_scalar(key, val, global_step + 1)

                info.update({
                    'global_step': global_step + 1,
                    'epoch': epoch + 1
                })

                wandb.log(info)

        epoch_simple_loss = 0
        epoch_vlb_loss = 0
        for step, (policies, measures, _) in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch_size = measures.shape[0]

            measures = measures.type(torch.float32).to(device)

            if cfg.latent_diffusion:
                with torch.no_grad():
                    batch = autoencoder.encode(policies).sample().detach()
                    # rescale the embeddings to be unit variance -- on first batch only
                    if epoch == 0 and step == 0:
                        print("Calculating scale factor...")
                        std = batch.flatten().std()
                        scale_factor = 1. / std
                        cfg.scale_factor = scale_factor.item()
                    batch *= scale_factor

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            losses, loss_dict, info_dict = gauss_diff.compute_training_losses(model, batch, t, model_kwargs={'cond': measures})
            loss = losses.mean()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if step % 100 == 0:
                log.info(f'grad norm: {grad_norm(model)}')
            optimizer.step()
            global_step += 1

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
        log.debug(f'Epoch: {epoch} Simple loss: {epoch_simple_loss / len(train_dataloader)}')
        if cfg.use_wandb:
            wandb.log({
                'losses/simple_loss': epoch_simple_loss / len(train_dataloader),
                'losses/vlb_loss': epoch_vlb_loss / len(train_dataloader),
                'epoch': epoch + 1,
                'global_step': global_step + 1
            })

    print('Saving final model checkpoint...')
    cp_name = f'diffusion_model_{cfg.env_name}_{datetime.now().strftime("%Y%m%d-%H%M")}.pt'
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), cp_name))
    # save the cfg
    save_cfg(model_checkpoint_folder, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)