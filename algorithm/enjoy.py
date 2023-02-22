import torch
import matplotlib.pyplot as plt

from models.unet import Unet
from autoencoders.conv_autoencoder import AutoEncoder
from diffusion.gaussian_diffusion import cosine_beta_schedule, linear_beta_schedule, GaussianDiffusion
from diffusion.latent_diffusion import LatentDiffusion


def visualize_generated_images(model_path, autoencoder_path):
    image_size = 28
    channels = 1
    timesteps = 600
    latent_channels = 64
    latent_size = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Unet(
        dim=latent_size,
        channels=latent_channels,
        dim_mults=(1, 2, 4,),
        use_convnext=True,
    )

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    autoencoder.to(device)
    autoencoder.eval()

    betas = cosine_beta_schedule(timesteps)
    gauss_diff = LatentDiffusion(betas, num_timesteps=timesteps, device=device)

    random_idx = torch.randint(0, 64, (1,))
    print(f'{random_idx=}')

    # sample 64 images
    latent_samples = gauss_diff.sample(model, latent_size, batch_size=64, channels=latent_channels)
    latent_samples = latent_samples[-1].to(device)

    samples = autoencoder.decode(latent_samples).detach().cpu().numpy()

    plt.imshow(samples[random_idx].reshape(image_size, image_size, channels), cmap="gray")
    plt.show()


if __name__ == '__main__':
    model_cp = './checkpoints/model_cp.pt'
    autoencoder_cp = './checkpoints/autoencoder.pt'
    visualize_generated_images(model_cp, autoencoder_cp)
