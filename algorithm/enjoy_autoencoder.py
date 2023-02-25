import torch
import matplotlib.pyplot as plt

from autoencoders.transformer_autoencoder import AutoEncoder
from dataset.mnist_fashion_dataset import transformed_dataset


def enjoy():
    model_cp = './checkpoints/autoencoder.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoEncoder(emb_channels=16, z_channels=8)
    model.load_state_dict(torch.load(model_cp))
    model.to(device)
    model.eval()

    batch = transformed_dataset['test']
    random_idx = torch.randint(0, 10000, (1,))
    input = batch[random_idx]['pixel_values'][0]
    input = input.to(device)
    output, _ = model(input.unsqueeze(0))
    output = output.detach().cpu()
    input = input.detach().cpu()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(input.reshape(32, 32, 1), cmap='gray')
    ax2.imshow(output.reshape(32, 32, 1), cmap='gray')
    plt.show()


if __name__ == '__main__':
    enjoy()