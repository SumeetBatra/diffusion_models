import torch
import numpy as np
import os

from pathlib import Path
from torch.optim import AdamW
from torchvision.utils import save_image
from models.unet import num_to_groups, Unet
from dataset.pytorch_dataset import dataloader
from diffusion.gaussian_diffusion import GaussianDiffusion, cosine_beta_schedule, linear_beta_schedule


def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def train():
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    save_and_sample_every = 1000

    model_checkpoint_folder = Path('./checkpoints')
    model_checkpoint_folder.mkdir(exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_size = 28
    channels = 1
    batch_size = 128

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,),
        use_convnext=True
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    epochs = 6

    timesteps = 600
    betas = cosine_beta_schedule(timesteps)
    gauss_diff = GaussianDiffusion(betas, num_timesteps=timesteps)
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch['pixel_values'].shape[0]
            batch = batch['pixel_values'].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            # loss = p_losses(model, batch, t, gauss_diff, loss_type='huber')
            losses = gauss_diff.compute_training_losses(model, batch, t)
            loss = losses.mean()


            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if step % 100 == 0:
                print(f'Loss: {loss.item()}')
                print(f'grad norm: {grad_norm(model)}')
            optimizer.step()

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


if __name__ == '__main__':
    train()