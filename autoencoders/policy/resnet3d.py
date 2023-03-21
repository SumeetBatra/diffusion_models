import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import List
from autoencoders.autoencoder_base import AutoEncoderBase

def normalization(channels: int):
    return nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-6)


def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        return self.nin_shortcut(x) + h


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels,  channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, 1, 0, 1, 0, 1), mode='constant', value=0)
        return self.conv(x)


class DownSample2D(nn.Module):
    '''Downsample the channels only, not the depth'''
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=(1, 2, 2), padding=(1, 0, 0))

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        return self.conv(x)


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

        self.conv_in = nn.Conv3d(in_channels, channels, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=1)

        # Number of channels in each top level block
        channels_list = [m * channels for m in [1] + channel_multipliers]

        self.down = nn.ModuleList()
        # Create top-level blocks
        for i in range(n_resolutions):
            # Each top level block consists of multiple ResNet Blocks and down-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]
            # Top-level block
            down = nn.Module()
            down.block = resnet_blocks
            if i != n_resolutions - 1:
                down.downsample = DownSample(channels) if i < 2 else DownSample2D(channels)
            else:
                down.downsample = nn.Identity()
            self.down.append(down)

        # Map to embedding space with a 4x4 convolution
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, 2 * z_channels, 3, stride=1, padding=1)

    def forward(self, policy: torch.Tensor):
        # Map to `channels` with the initial convolution
        x = self.conv_in(policy)

        # Top-level blocks
        for i, down in enumerate(self.down):
            # ResNet Blocks
            for block in down.block:
                x = block(x)
            # Down-sampling
            x = down.downsample(x)

        # temp remove the depth dim for attn to work
        x = x.squeeze(dim=2)
        # Normalize and map to embedding space
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):
    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int,
                 out_channels: int, z_channels: int):
        """
        :param channels: is the number of channels in the final convolution layer
        :param channel_multipliers: are the multiplicative factors for the number of channels in the
            previous blocks, in reverse order
        :param n_resnet_blocks: is the number of resnet layers at each resolution
        :param out_channels: is the number of channels in the image
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()
        self.channels = channels

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        num_resolutions = len(channel_multipliers)

        # Number of channels in each top level block, in the reverse order
        channels_list = [m * channels for m in channel_multipliers]

        # Number of channels in the  top-level block
        channels = channels_list[-1]

        # Initial $3 \times 3$ convolution layer that maps the embedding space to channels and depth
        self.conv_in = nn.Conv2d(z_channels, 2 * channels, 3, stride=1, padding=1)

        # List of top-level blocks
        self.up = nn.ModuleList()
        # Create top-level blocks
        for i in reversed(range(num_resolutions)):
            # Each top level block consists of multiple ResNet Blocks and up-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            for _ in range(n_resnet_blocks + 1):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                channels = channels_list[i]
            # Top-level block
            up = nn.Module()
            up.block = resnet_blocks
            # Conv3DTranspose Up-sampling at the end of each top level block except the first
            if i != 0:
                up.upsample = nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2)
            else:
                up.upsample = nn.Identity()
            # Prepend to be consistent with the checkpoint
            self.up.insert(0, up)

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, out_channels, 3, stride=1, padding=(0, 1, 1))

    def forward(self, z: torch.Tensor):
        """
        :param z: is the embedding tensor with shape `[batch_size, z_channels, z_height, z_height]`
        """

        # Map to `channels` with the initial convolution
        h = self.conv_in(z)

        # add the depth dim back in
        h = rearrange(h, "b (d c) h w -> b d c h w", d=2, c=2)

        # Top-level blocks
        for up in reversed(self.up):
            # ResNet Blocks
            for block in up.block:
                h = block(h)
            # Up-sampling
            h = up.upsample(h)

        # Normalize and map to image space
        h = self.norm_out(h)
        h = swish(h)
        policy = self.conv_out(h)

        #
        return policy


class ResNet3DAutoEncoder(AutoEncoderBase):
    def __init__(self, emb_channels: int, z_channels: int):
        """
        :param emb_channels: is the number of dimensions in the quantized embedding space
        :param z_channels: is the number of channels in the embedding space
        """
        AutoEncoderBase.__init__(self, emb_channels, z_channels)

        self.encoder = Encoder(channels=2, channel_multipliers=[1, 2, 4], n_resnet_blocks=3, in_channels=1, z_channels=z_channels)
        self.decoder = Decoder(channels=2, channel_multipliers=[4, 2, 1], n_resnet_blocks=3, out_channels=1, z_channels=z_channels)
