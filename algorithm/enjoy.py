import torch
import matplotlib.pyplot as plt

from models.unet import Unet
from autoencoders.conv_autoencoder import AutoEncoder
from diffusion.gaussian_diffusion import cosine_beta_schedule, linear_beta_schedule, GaussianDiffusion
from diffusion.latent_diffusion import LatentDiffusion


def visualize_generated_images(model_path, autoencoder_path):
    latent_diffusion = False
    image_size = 28
    channels = 1
    timesteps = 600
    latent_channels = 64
    latent_size = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    betas = cosine_beta_schedule(timesteps)

    if latent_diffusion:
        model = Unet(
            dim=latent_size,
            channels=latent_channels,
            dim_mults=(1, 2, 4,),
            use_convnext=True,
        )
        autoencoder = AutoEncoder()
        autoencoder.load_state_dict(torch.load(autoencoder_path))
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

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    random_idx = torch.randint(0, 64, (1,))
    print(f'{random_idx=}')

    size, channels = (latent_size, latent_channels) if latent_diffusion else (image_size, channels)

    # sample 64 images
    samples = gauss_diff.sample(model, size, batch_size=64, channels=channels)

    if latent_diffusion:
        samples = samples[-1].to(device)
        samples = autoencoder.decode(samples).detach().cpu().numpy()
    else:
        samples = samples[-1]

    plt.imshow(samples[random_idx].reshape(image_size, image_size, channels), cmap="gray")
    plt.show()


if __name__ == '__main__':
    model_cp = './checkpoints/model_cp.pt'
    autoencoder_cp = './checkpoints/autoencoder.pt'
    visualize_generated_images(model_cp, autoencoder_cp)
