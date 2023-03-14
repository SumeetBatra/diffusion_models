#  https://nn.labml.ai/diffusion/stable_diffusion/model/autoencoder.html
import torch
import torch.nn.functional as F

from torch import nn
from typing import List

from RL.actor_critic import Actor
import numpy as np
from hyper.ghn_modules import *

def normalization(channels: int):
    return nn.GroupNorm(num_groups=1, num_channels=channels, eps=1e-6)


def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
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


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        return self.conv(x)


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = normalization(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)

        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(b, c, h, w)
        out = self.proj_out(out)
        return x + out


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


class HyperAutoEncoder(nn.Module):
    """
    ## Autoencoder

    This consists of the encoder and decoder modules.
    """

    def __init__(self, actor_cfg, emb_channels: int, z_channels: int):
        """
        :param emb_channels: is the number of dimensions in the quantized embedding space
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()
        # self.encoder = Encoder(channels=2, channel_multipliers=[1, 2, 4, 8], n_resnet_blocks=3, in_channels=1, z_channels=z_channels)
        self.encoder = ModelEncoder(actor_cfg, z_channels)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.decoder = Decoder(channels=2, channel_multipliers=[8, 4, 2, 1], n_resnet_blocks=3, out_channels=1, z_channels=z_channels)
        config = {}
        config['max_shape'] = (256, 256, 1, 1)
        config['num_classes'] = 2 * actor_cfg['action_shape'][0]
        config['num_observations'] = actor_cfg['obs_shape'][0]
        config['weight_norm'] = True
        config['ve'] = 1 > 1
        config['layernorm'] = True
        config['hid'] = 16
        self.decoder = MLP_GHN(**config,
                    debug_level=0, device=device).to(device)          
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)

        def make_actor():
            return Actor(actor_cfg, obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape, deterministic=True)
        self.dummy_actor = make_actor

    def encode(self, img: torch.Tensor) -> 'GaussianDistribution':
        """
        ### Encode images to latent representation

        :param img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(img)
        # Get the moments in the quantized embedding space
        # moments = self.quant_conv(z)
        # Return the distribution
        # return GaussianDistribution(moments)
        return z

    def decode(self, z: torch.Tensor):
        """
        ### Decode images from latent representation

        :param z: is the latent representation with shape `[batch_size, emb_channels, z_height, z_height]`
        """
        # Map to embedding space from the quantized representation
        # z = self.post_quant_conv(z)
        # Decode the image of shape `[batch_size, channels, height, width]`
        return self.decoder([ self.dummy_actor() for _ in range(z.shape[0])], z)

    def forward(self, x: torch.Tensor):
        # posterior = self.encode(x)
        moment = self.encode(x)
        # z = posterior.sample()
        out = self.decode(moment)
        # out = self.decode(posterior.mean)
        # return out, posterior
        return out
    
    def to(self, device):
        super().to(device)
        self.encoder.to(device)


class ModelEncoder(nn.Module):
    def __init__(self, actor_cfg, z_channels):
        super().__init__()
        dummy_actor = Actor(actor_cfg, obs_shape=actor_cfg.obs_shape[0], action_shape=actor_cfg.action_shape)
              

        self.channels = [1, 32, 64, 64, 128, 256, 256, 512, 512, 512]
        self.kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
        self.strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

        self.max_pool_kernel_sizes = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        self.max_pool_strides = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1]

        self.cnns = {}
        total_op_shape = 0
        self.list_of_weight_names = []
        self.list_of_real_weight_names = []
        for name, param in dummy_actor.named_parameters():

            self.list_of_real_weight_names.append(name)
            key_name = "_".join(name.split('.'))
            self.list_of_weight_names.append(key_name)

            if 'weight' in name:
                shape = (1,1,)+tuple(param.data.shape)
                self.cnns[key_name], op_shape = self._create_cnn_backbone(shape)
            
            elif 'bias' in name:
                shape = (1,1,) + tuple(param.data.shape) + (1,)
                self.cnns[key_name], op_shape = self._create_fc_backbone(shape)

            else:
                shape = (1,)+tuple(param.data.shape) + (1,)
                self.cnns[key_name], op_shape = self._create_fc_backbone(shape)

            total_op_shape += np.prod(op_shape)
        
        self.cnns = nn.ModuleDict(self.cnns)

        self.out = nn.Linear(total_op_shape, 8*4*z_channels)

        
    # create a cnn backbone to extract features from tensor of shape (batch_size, *shape)        
    def _create_cnn_backbone(self, shape, leaky_relu=False):

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = self.channels[i]
            output_channel = self.channels[i+1]


            cnn_block = nn.Sequential()
            cnn_block.add_module(
                f'conv{i}',
                nn.Conv2d(input_channel, output_channel, self.kernel_sizes[i], self.strides[i], self.paddings[i])
            )

            if batch_norm:
                cnn_block.add_module(f'batchnorm{i}', nn.BatchNorm2d(output_channel))

            relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
            cnn_block.add_module(f'relu{i}', relu)   
            cnn_block.add_module(f'maxpool_{i}', nn.MaxPool2d(self.max_pool_kernel_sizes[i], self.max_pool_strides[i]))
            
            cnn.add_module(f'cnn_block_{i}', cnn_block)


        def get_output_shape(input_shape, channel, kernel_size, stride, padding, pool_size=2, pool_stride=2):
            # input shape can be rectangular
            dim1 = (input_shape[2] - kernel_size + 2*padding) // stride + 1
            dim2 = (input_shape[3] - kernel_size + 2*padding) // stride + 1
            dim1 = (dim1 - pool_size) // pool_stride + 1
            dim2 = (dim2 - pool_size) // pool_stride + 1
            return (1, channel, int(dim1), int(dim2))

        for i in range(len(self.channels)-2):
            channel = self.channels[i+1]
            kernel_size = self.kernel_sizes[i]
            stride = self.strides[i]
            padding = self.paddings[i]
            pool_size = self.max_pool_kernel_sizes[i]
            pool_stride = self.max_pool_strides[i]

            output_shape = get_output_shape(shape, channel, kernel_size, stride, padding, pool_size, pool_stride)

            if min(output_shape) > 0:
                conv_relu(i, batch_norm=True)
                shape = output_shape
            else:
                break
        return cnn, shape

    def _create_fc_backbone(self, shape):
        fc = nn.Sequential()
        fc.add_module('fc1', nn.Linear(shape[2], 256))
        fc.add_module('relu1', nn.ReLU(inplace=True))
        fc.add_module('fc2', nn.Linear(256, 256))
        fc.add_module('relu2', nn.ReLU(inplace=True))
        fc.add_module('fc3', nn.Linear(256, 256))
        fc.add_module('relu3', nn.ReLU(inplace=True))
        return fc, (1, 1, 256, 1)

    def forward(self, x):
        outs = []
        for k in range(len(self.list_of_weight_names)):
            name = self.list_of_weight_names[k]
            real_name = self.list_of_real_weight_names[k]
            if 'weight' in name:
                out = self.cnns[name](x[real_name].unsqueeze(1))
                out = out.view(out.size(0), -1)
                outs.append(out)
            elif 'bias' in name:
                out = self.cnns[name](x[real_name].unsqueeze(1))
                out = out.view(out.size(0), -1)
                outs.append(out)
            else:
                out = self.cnns[name](x[real_name].unsqueeze(1))
                out = out.view(out.size(0), -1)
                outs.append(out)



        x = torch.cat(outs, dim=1)
        x = self.out(x)
        return x.reshape(-1,8,4,4)

    def to(self, device):
        # self.dummy_actor.to(device)
        for cnn in self.cnns.values():
            cnn.to(device)
        self.out.to(device)
        return self

