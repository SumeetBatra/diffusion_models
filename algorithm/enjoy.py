import torch
import matplotlib.pyplot as plt

from models.unet import Unet
from diffusion.gaussian_diffusion import cosine_beta_schedule, linear_beta_schedule, GaussianDiffusion

def visualize_generated_images(model_path):
    image_size = 28
    channels = 1
    batch_size = 128
    timesteps = 600

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    betas = cosine_beta_schedule(timesteps)
    gauss_diff = GaussianDiffusion(betas, num_timesteps=timesteps)

    random_idx = torch.randint(0, 64, (1,))
    print(f'{random_idx=}')

    # sample 64 images
    samples = gauss_diff.sample(model, image_size, batch_size=64, channels=channels)

    plt.imshow(samples[-1][random_idx].reshape(image_size, image_size, channels), cmap="gray")
    plt.show()



if __name__ == '__main__':
    model_cp = './checkpoints/model_cp.pt'
    visualize_generated_images(model_cp)