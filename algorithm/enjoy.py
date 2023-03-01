import torch
import matplotlib.pyplot as plt
import json

from attrdict import AttrDict
from models.unet import Unet
from autoencoders.transformer_autoencoder import AutoEncoder
from diffusion.gaussian_diffusion import cosine_beta_schedule, linear_beta_schedule, GaussianDiffusion
from diffusion.latent_diffusion import LatentDiffusion
from diffusion.ddim import DDIMSampler


def visualize_generated_images(model_path, autoencoder_path):
    latent_diffusion = True
    use_ddim = True
    image_size = 32
    channels = 1
    timesteps = 600
    latent_channels = 8
    latent_size = 4

    cfg_path = './checkpoints/cfg.json'
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
        cfg = AttrDict(cfg)
    scale_factor = cfg.scale_factor if latent_diffusion else None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    betas = cosine_beta_schedule(timesteps)

    autoencoder = None
    if latent_diffusion:
        logvar = torch.full(fill_value=0., size=(timesteps,))
        model = Unet(
            dim=64,
            channels=latent_channels,
            dim_mults=(1, 2, 4,),
            logvar=logvar
        )
        autoencoder = AutoEncoder(emb_channels=8, z_channels=4)
        autoencoder.load_state_dict(torch.load(autoencoder_path))
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

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    ddim_sampler = DDIMSampler(gauss_diff, n_steps=50)

    random_idx = torch.randint(0, 64, (1,))
    print(f'{random_idx=}')

    size, channels = (latent_size, latent_channels) if latent_diffusion else (image_size, channels)

    # sample 64 images
    if use_ddim:
        samples = ddim_sampler.sample(model, shape=[64, channels, size, size], cond=None)
    else:
        samples = gauss_diff.sample(model, size, batch_size=64, channels=channels)

    if latent_diffusion:
        if not use_ddim:
            samples = samples[-1].to(device)
            # rescale to be in distribution of the autoencoder's latent space
        samples = samples * (1. / scale_factor)
        samples = autoencoder.decode(samples).detach().cpu().numpy()
    else:
        # standard diffusion
        samples = samples[-1]

    plt.imshow(samples[random_idx].reshape(image_size, image_size, 1), cmap="gray")
    plt.show()


if __name__ == '__main__':
    model_cp = './checkpoints/model_cp.pt'
    autoencoder_cp = './checkpoints/autoencoder.pt'
    visualize_generated_images(model_cp, autoencoder_cp)
