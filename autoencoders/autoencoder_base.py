import torch
import torch.nn as nn

from abc import abstractmethod


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


class AutoEncoderBase(nn.Module):
    def __init__(self, emb_channels: int, z_channels: int, conditional: bool = False):
        super().__init__()
        self.encoder: nn.Module = None
        self.decoder: nn.Module = None
        self.conditional = conditional

        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, x: torch.Tensor, y: torch.Tensor = None):
        assert self.encoder is not None, "Need to define a valid encoder (nn.Module)"
        z = self.encoder(x,y)
        moments = self.quant_conv(z)
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor, y: torch.Tensor = None):
        assert self.decoder is not None, "Need to define a valid decoder (nn.Module)"
        z = self.post_quant_conv(z)
        return self.decoder(z, y)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        posterior = self.encode(x, y)
        z = posterior.sample()
        out = self.decode(z, y)
        return out, posterior
