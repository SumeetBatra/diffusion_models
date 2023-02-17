import torch
import matplotlib.pyplot as plt

from autoencoders.conv_autoencoder import AutoEncoder
from dataset.mnist_fashion_dataset import transformed_dataset

def enjoy():
    model_cp = './checkpoints/autoencoder.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoEncoder()
    model.load_state_dict(torch.load(model_cp))
    model.to(device)
    model.eval()

    batch = transformed_dataset['test']
    random_idx = torch.randint(0, 10000, (1,))
    input = batch[random_idx]['pixel_values'][0]
    input = input.to(device)
    output = model(input).detach().cpu()
    input = input.detach().cpu()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(input.reshape(28, 28, 1), cmap='gray')
    ax2.imshow(output.reshape(28, 28, 1), cmap='gray')
    plt.show()


if __name__ == '__main__':
    enjoy()