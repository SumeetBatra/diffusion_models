import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from autoencoders.autoencoder_base import AutoEncoderBase


class LayerEmbedder(nn.Module):
    def __init__(self, max_layer_size: int = 128):
        super().__init__()
        self.weight_embedder = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.bias_embedder = nn.Linear(max_layer_size, max_layer_size)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]):
        weight, bias = x
        bias = bias[0]
        w_out = self.weight_embedder(weight)
        b_out = self.bias_embedder(bias)
        out = 0

class PolicyEmbedder(nn.Module):
    def __init__(self, max_layer_size: int = 128, num_layers: int = 3):
        '''Create an embedding of the policy layer by layer'''
        super().__init__()
        self.embedder = nn.ModuleList()
        for i in range(num_layers):
            self.embedder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
                    nn.Linear(max_layer_size, max_layer_size)
                )
            )

    def forward(self, policy_tensor: torch.Tensor):
        for i, embed in enumerate(self.embedder):
            layer, bias = policy_tensor[2 * i], policy_tensor[2 * i + 1]


class Encoder(nn.Module):
    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int,
                 in_channels: int, z_channels: int):
        """
        :param channels: is the number of channels in the first convolution layer
        :param channel_multipliers: are the multiplicative factors for the number of channels in the
            subsequent blocks
        :param n_resnet_blocks: is the number of resnet layers at each resolution
        :param in_channels: is the number of channels in the image
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()
        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        n_resolutions = len(channel_multipliers)

