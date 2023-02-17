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
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, 3),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        z_channels = 32
        quant_emb_channels = 2 * z_channels
        # map from embeddings to quantized embedding space moments (mean and variance)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * quant_emb_channels, 1)
        # map from quantized embeddings to embedding space
        self.post_quant_conv = nn.Conv2d(quant_emb_channels, z_channels, 1)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mean()
        dec = self.decode(z)
        return dec, posterior

    def encode(self, x):
        z = self.encoder(x)
        moments = self.quant_conv(z)
        return GaussianDistribution(moments)

    def decode(self, x):
        z = self.post_quant_conv(x)
        return self.decoder(z)


class GaussianDistribution:
    # https://nn.labml.ai/diffusion/stable_diffusion/model/autoencoder.html
    def __init__(self, parameters: torch.Tensor):
        self.mean, logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.var = torch.exp(self.logvar)
        self.std = torch.exp(0.5 * self.logvar)
        self.deterministic = False

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.std)

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])