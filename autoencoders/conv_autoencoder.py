# a simple convolutional autoencoder for the MNIST fashion dataset
import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU()
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.proj(x)
        h = self.norm(h)
        h = self.act(h)

        return h + self.res_conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ResnetBlock(in_channels=1, out_channels=2),
            Downsample(in_channels=2),
            ResnetBlock(in_channels=2, out_channels=4),
            Downsample(in_channels=4),
            ResnetBlock(in_channels=4, out_channels=8),
            Downsample(in_channels=8),
            ResnetBlock(in_channels=8, out_channels=8)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ResnetBlock(in_channels=4, out_channels=4),
            Upsample(in_channels=4),
            ResnetBlock(in_channels=4, out_channels=2),
            Upsample(in_channels=2),
            ResnetBlock(in_channels=2, out_channels=1),
            Upsample(in_channels=1),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.layers(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        z_channels = 4
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
            z = posterior.mean
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