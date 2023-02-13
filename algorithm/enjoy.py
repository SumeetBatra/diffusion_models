import torch
import matplotlib.pyplot as plt

from sampling.sampler import sample
from models.unet import Unet
from diffusion.forward_diffusion import cosine_beta_schedule, linear_beta_schedule, get_vars_from_schedule

def visualize_generated_images(model_path):
    image_size = 28
    channels = 1
    batch_size = 128
    timesteps = 300

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    betas, sqrt_alphas_cumprod, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance = get_vars_from_schedule(
        linear_beta_schedule, timesteps=timesteps)

    random_idx = torch.randint(0, 64, (1,))
    print(f'{random_idx=}')

    # sample 64 images
    samples = sample(model, image_size, timesteps, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, batch_size=64, channels=channels)

    plt.imshow(samples[-1][random_idx].reshape(image_size, image_size, channels), cmap="gray")
    plt.show()



if __name__ == '__main__':
    model_cp = './checkpoints/model_cp.pt'
    visualize_generated_images(model_cp)