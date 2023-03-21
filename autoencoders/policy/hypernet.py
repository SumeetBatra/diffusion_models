import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders.policy.resnet3d import Encoder as Resnet3DEncoder
from autoencoders.autoencoder_base import AutoEncoderBase
from models.hyper.ghn_modules import *
from RL.actor_critic import Actor


class HypernetAutoEncoder(AutoEncoderBase):
    def __init__(self, emb_channels: int, z_channels: int):
        """
        :param emb_channels: is the number of dimensions in the quantized embedding space
        :param z_channels: is the number of channels in the embedding space
        """
        AutoEncoderBase.__init__(self, emb_channels, z_channels)

        self.encoder = Resnet3DEncoder(channels=2, channel_multipliers=[1, 2, 4, 8, 8, 8], n_resnet_blocks=4, in_channels=1,
                                       z_channels=z_channels)

        # config dict for the hypernet decoder
        action_dim, obs_dim = 6, 18
        config = {}
        config['max_shape'] = (256, 256, 1, 1)
        config['num_classes'] = 2 * action_dim
        config['num_observations'] = obs_dim
        config['weight_norm'] = True
        config['ve'] = 1 > 1
        config['layernorm'] = True
        config['hid'] = 16
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = MLP_GHN(**config, debug_level=0, device=device)

        def make_actor():
            return Actor(obs_shape=18, action_shape=np.array([action_dim]), deterministic=True)

        self.dummy_actor = make_actor

    def decode(self, z: torch.Tensor):
        """
        ### Decode policies from latent representation
        :param z: is the latent representation with shape `[batch_size, emb_channels, z_height, z_height]`
        """
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)
        # Decode the image of shape `[batch_size, channels, height, width]`
        return self.decoder([self.dummy_actor() for _ in range(z.shape[0])], z)
