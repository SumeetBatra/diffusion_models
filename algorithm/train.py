import torch
import numpy as np
import os
import argparse
import wandb
import matplotlib.pyplot as plt

from attrdict import AttrDict
from distutils.util import strtobool
from pathlib import Path
from utils.utilities import config_wandb, save_cfg
from torch.optim import AdamW
from torchvision.utils import save_image
from models.unet import num_to_groups, Unet
from autoencoders.transformer_autoencoder import AutoEncoder
from dataset.mnist_fashion_dataset import dataloader
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

    if cfg.use_wandb:
        config_wandb(run_name=cfg.wandb_run_name, wandb_project=cfg.wandb_project, wandb_group=cfg.wandb_group,
                     cfg=cfg)

    autoencoder_checkpoint_path = Path('./checkpoints/autoencoder.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_size = 32
    channels = 1
    batch_size = 128

    timesteps = 600
    betas = cosine_beta_schedule(timesteps)

    autoencoder = None
    if cfg.latent_diffusion:
        latent_channels = 16
        latent_size = 4

        logvar = torch.full(fill_value=0., size=(timesteps,))
        model = Unet(
            dim=latent_size,
            channels=latent_channels,
            dim_mults=(1, 2, 4,),
            use_convnext=True,
            logvar=logvar
        )
        autoencoder = AutoEncoder(emb_channels=16, z_channels=8)
        autoencoder.load_state_dict(torch.load(str(autoencoder_checkpoint_path)))
        autoencoder.to(device)
        autoencoder.eval()

        gauss_diff = LatentDiffusion(betas, num_timesteps=timesteps, device=device)
    else:
        model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4),
            use_convnext=True,
            out_dim=2 * channels
        )
        gauss_diff = GaussianDiffusion(betas, num_timesteps=timesteps, device=device)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    epochs = 10
    scale_factor = 1.0
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch['pixel_values'].shape[0]
            batch = batch['pixel_values'].to(device)

            if cfg.latent_diffusion:
                with torch.no_grad():
                    batch = autoencoder.encode(batch).sample().detach()
                    # rescale the embeddings to be unit variance -- on first batch only
                    if epoch == 0 and step == 0:
                        print("Calculating scale factor...")
                        std = batch.flatten().std()
                        scale_factor = 1. / std
                        cfg.scale_factor = scale_factor.item()
                    batch *= scale_factor

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            losses, info_dict = gauss_diff.compute_training_losses(model, batch, t)
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

            # save generated images
            if step != 0 and step % save_and_sample_every == 0:
                print('Sampling...')
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: gauss_diff.sample(model,
                                                            image_size=image_size,
                                                            batch_size=n,
                                                            channels=channels),
                                           batches))
                all_images = torch.cat(all_images_list[0], dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)

    print('Saving final model checkpoint...')
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), 'model_cp.pt'))
    # save the cfg
    save_cfg(model_checkpoint_folder, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)