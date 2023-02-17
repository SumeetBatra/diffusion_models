import torch
import numpy as np
import os

from pathlib import Path
from torch.optim import Adam
from autoencoders.conv_autoencoder import AutoEncoder
from dataset.mnist_fashion_dataset import dataloader


def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def train_autoencoder():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoEncoder()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    loss_func = torch.nn.MSELoss()

    model_checkpoint_folder = Path('./checkpoints')
    model_checkpoint_folder.mkdir(exist_ok=True)

    epochs = 10
    for epoch in range(epochs):
        print(f'{epoch=}')

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch['pixel_values'].shape[0]
            batch = batch['pixel_values'].to(device)

            out = model(batch)
            loss = loss_func(batch, out)

            loss.backward()
            if step % 100 == 0:
                print(f'Loss: {loss.item()}')
                print(f'grad norm: {grad_norm(model)}')
            optimizer.step()

    print('Saving final model checkpoint...')
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), 'autoencoder.pt'))


if __name__ == '__main__':
    train_autoencoder()

