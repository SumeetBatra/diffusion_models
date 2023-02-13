import torch

from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image
from models.unet import num_to_groups, Unet
from dataset.pytorch_dataset import dataloader
from losses.loss_functions import p_losses
from diffusion.forward_diffusion import get_vars_from_schedule, cosine_beta_schedule
from sampling.sampler import sample


def train():
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    save_and_sample_every = 1000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_size = 28
    channels = 1
    batch_size = 128

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = 6

    timesteps = 300
    sqrt_alphas_cumprod, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance = get_vars_from_schedule(cosine_beta_schedule, timesteps=timesteps)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch['pixel_values'].shape[0]
            batch = batch['pixel_values'].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, loss_type='huber')

            if step % 100 == 0:
                print(f'Loss: {loss.item()}')

            loss.backward()
            optimizer.step()

            # save generated images
            # save generated images
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)









if __name__ == '__main__':
    train()