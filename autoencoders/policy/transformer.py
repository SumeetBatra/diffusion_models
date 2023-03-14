import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders.transformer_autoencoder import Encoder as TransformerEncoder
from autoencoders.transformer_autoencoder import GaussianDistribution
from autoencoders.transformer_autoencoder import Decoder as TransformerDecoder


class TransformerPolicyAutoencoder(nn.Module):
    '''Autoencoder for NN policies'''
    def __init__(self, emb_channels: int, z_channels: int):
        """
        :param emb_channels: is the number of dimensions in the quantized embedding space
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()
        self.encoder = TransformerEncoder(channels=2, channel_multipliers=[1, 2, 4, 8], n_resnet_blocks=3, in_channels=1, z_channels=z_channels, use_conv3d=True)

        self.decoder = TransformerDecoder(channels=2, channel_multipliers=[8, 4, 2, 1], n_resnet_blocks=3, out_channels=1, z_channels=z_channels, use_conv3d=True)

        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, x: torch.Tensor) -> GaussianDistribution:
        '''
        Encode padded policies into a latent representation
        :param x: policies tensor with shape [batch_size, num_layers, layer_height, layer_width]
        where layer_height = layer_width = size of the largest hidden layer in the network
        '''

        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(x)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        # Return the distribution
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        '''
        Decode policies from latent representation
        :param z: is the latent representation with shape `[batch_size, emb_channels, z_height, z_height]`
        '''

        # map to embedding space from a quantized representation
        z = self.post_quant_conv(z)
        # decode the policies
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        posterior = self.encode(x)
        z = posterior.sample()
        out = self.decode(z)
        return out, posterior