# a simple convolutional autoencoder for the MNIST fashion dataset
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x