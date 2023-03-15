import torch
import numpy as np
import os
import pickle
import glob

from pathlib import Path
from torch.optim import Adam
from dataset.policy_dataset import ElitesDataset
from torch.utils.data import DataLoader
from autoencoders.policy.resnet3d import ResNet3DAutoEncoder
from autoencoders.policy.transformer import TransformerPolicyAutoencoder
from RL.actor_critic import Actor


def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def dataset_factory():
    archive_data_path = '/home/sumeet/diffusion_models/data'
    archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            archive_df = pickle.load(f)
            archive_dfs.append(archive_df)

    mlp_shape = (128, 128, 6)

    dummy_agent = Actor(obs_shape=18, action_shape=np.array([6]))

    elite_dataset = ElitesDataset(archive_dfs, mlp_shape, dummy_agent)

    return DataLoader(elite_dataset, batch_size=32, shuffle=True)


def train_autoencoder():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_checkpoint = None
    model = TransformerPolicyAutoencoder(emb_channels=64, z_channels=32)
    if model_checkpoint is not None:
        print(f'Loading model from checkpoint')
        model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    mse_loss_func = torch.nn.MSELoss()
    kl_loss_coef = 1e-4

    model_checkpoint_folder = Path('./checkpoints')
    model_checkpoint_folder.mkdir(exist_ok=True)

    dataloader = dataset_factory()

    disc_start = 50001
    kl_weight = 1e-6
    disc_weight = 0.5
    # loss_func = LPIPSWithDiscriminator(disc_start, kl_weight=kl_weight, disc_weight=disc_weight)
    # optimizer2 = Adam(loss_func.discriminator.parameters(), lr=1e-3)

    epochs = 20
    global_step = 0
    for epoch in range(epochs):
        print(f'{epoch=}')
        print(f'{global_step=}')

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.to(device)

            img_out, posterior = model(batch)
            # loss = loss_func(batch, img_out, posterior, global_step, 0)
            # loss += loss_func(batch, img_out, posterior, global_step, 1)
            loss = mse_loss_func(batch, img_out) + kl_loss_coef * posterior.kl().mean()

            loss.backward()
            if step % 100 == 0:
                print(f'Loss: {loss.item()}')
                print(f'grad norm: {grad_norm(model)}')
            optimizer.step()
            global_step += step

    print('Saving final model checkpoint...')
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), 'autoencoder.pt'))


if __name__ == '__main__':
    train_autoencoder()

