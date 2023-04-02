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
from utils.utilities import config_wandb, save_cfg
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models.unet import num_to_groups, Unet
from models.cond_unet import ConditionalUNet
from autoencoders.policy.hypernet import HypernetAutoEncoder as AutoEncoder
from dataset.policy_dataset import ShapedEliteDataset
from diffusion.gaussian_diffusion import GaussianDiffusion, cosine_beta_schedule, linear_beta_schedule
from diffusion.latent_diffusion import LatentDiffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_diffusion', type=lambda x: bool(strtobool(x)), default=True, help='Use latent diffusion or standard (improved) diffusion')
    # wandb args
    parser.add_argument('--use_wandb', type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_project', type=str, default='qd_diffusion')
    parser.add_argument('--wandb_run_name', type=str, default='diffusion_run')
    parser.add_argument('--wandb_group', type=str, default='diffusion_group')
    args = parser.parse_args()
    cfg = AttrDict(vars(args))
    return cfg


def shaped_elites_dataset_factory():
    archive_data_path = '/home/sumeet/diffusion_models/data'
    archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100_global*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            archive_df = pickle.load(f)
            archive_dfs.append(archive_df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    s_elite_dataset = ShapedEliteDataset(archive_dfs, obs_dim=18, action_shape=np.array([6]), device=device, normalize_obs=True, is_eval=False)

    return DataLoader(s_elite_dataset, batch_size=32, shuffle=True)


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
    model_checkpoint = None

    if cfg.use_wandb:
        config_wandb(run_name=cfg.wandb_run_name, wandb_project=cfg.wandb_project, wandb_group=cfg.wandb_group,
                     cfg=cfg)

    autoencoder_checkpoint_path = Path('./checkpoints/autoencoder_20230401-162949_autoencoder.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_size = 32
    channels = 1
    batch_size = 32

    timesteps = 600
    betas = cosine_beta_schedule(timesteps)

    autoencoder = None
    if cfg.latent_diffusion:
        latent_channels = 8
        latent_size = 8

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
        autoencoder = AutoEncoder(emb_channels=8, z_channels=4)
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

    dataloader = shaped_elites_dataset_factory()

    epochs = 40
    scale_factor = 1.0
    for epoch in range(epochs):
        for step, (policies, measures, _) in enumerate(dataloader):
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

            losses, info_dict = gauss_diff.compute_training_losses(model, batch, t, model_kwargs={'cond': measures})
            loss = losses.mean()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if step % 100 == 0:
                print(f'Loss: {loss.item()}')
                print(f'grad norm: {grad_norm(model)}')
            optimizer.step()

            # maybe log to wandb
            if cfg.use_wandb:
                wandb.log(info_dict)
                wandb.log({
                    'data/batch_mean': batch.mean().item(),
                    'data/batch_var': batch.var().item(),
                })

    print('Saving final model checkpoint...')
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), 'model_cp.pt'))
    # save the cfg
    save_cfg(model_checkpoint_folder, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)