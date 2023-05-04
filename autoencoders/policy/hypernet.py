import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders.policy.resnet3d import Encoder as Resnet3DEncoder
from autoencoders.autoencoder_base import AutoEncoderBase, GaussianDistribution
from models.hyper.ghn_modules import *
from RL.actor_critic import Actor


def classifier_gradients(classifier: nn.Module, x: torch.Tensor, y: torch.Tensor, classifier_scale: int = 1):
    # TODO: THIS WON'T WORK! Need to move this to a model that predicts measures from latent codes
    '''
    Gets the gradients of the error between the policies' target and predicted measures
    :param classifier: Model that predicts measures from latent codes
    :param x: batch of latent codes
    :param y: target measures
    :param classifier_scale: scale of the gradient magnitude. Higher will result in stronger conditioning
    '''
    with torch.enable_grad():
        x.detach().requires_grad_(True)
        pred = classifier(x)
        loss = F.mse_loss(pred, y)
        return torch.autograd.grad(loss, x)[0] * classifier_scale


class HypernetAutoEncoder(AutoEncoderBase):
    def __init__(self,
                 emb_channels: int,
                 z_channels: int,
                 obs_shape: int,
                 action_shape: np.ndarray,
                 z_height: int = 4,
                 conditional: bool = False,
                 ghn_hid: int = 64,
                 obsnorm_hid: int = 64,
                 enc_fc_hid: int = 64,
                 ):
        """
        :param emb_channels: is the number of dimensions in the quantized embedding space
        :param z_channels: is the number of channels in the embedding space
        """
        AutoEncoderBase.__init__(self, emb_channels, z_channels, z_height, conditional)
        # TODO: factor in the obs-norm params into the weights dict for the model encoder and decoder
        self.encoder = ModelEncoder(obs_shape=obs_shape,
                                    action_shape=action_shape,
                                    emb_channels=emb_channels,
                                    z_channels=z_channels,
                                    z_height=z_height,
                                    conditional=conditional,
                                    fc_hid=enc_fc_hid,
                                    )
        

        # config dict for the hypernet decoder
        action_dim, obs_dim = action_shape[0], obs_shape
        config = {}
        config['max_shape'] = (256, 256, 1, 1) if obs_dim > 128 else (128, 128, 1, 1)
        config['num_classes'] = 2 * action_dim
        config['num_observations'] = obs_dim
        config['weight_norm'] = False
        config['ve'] = 1 > 1
        config['layernorm'] = True
        config['hid'] = ghn_hid
        config['z_channels'] = z_channels
        config['z_height'] = z_height
        config['norm_variables'] = False
        config['conditional'] = conditional
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = MLP_GHN(**config, debug_level=0, device=device)

        self.obsnorm_decoder = ObsNormDecoder(obs_shape=obs_shape,
                                        z_channels=z_channels,
                                        z_height=z_height,
                                        conditional=conditional,
                                        hid = obsnorm_hid,
                                        )
        
        print(f"Total size of z is: {self.decoder.z_vec_size}")

        def make_actor():
            return Actor(obs_shape=obs_shape,
                         action_shape=action_shape,
                         deterministic=True,
                         normalize_obs=False)

        self.dummy_actor = make_actor

    def encode(self, x: torch.Tensor, y: torch.Tensor = None):
        assert self.encoder is not None, "Need to define a valid encoder (nn.Module)"
        z = self.encoder(x, y)
        moments = self.quant_conv(z)
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor, y: torch.Tensor = None):
        """
        ### Decode policies from latent representation
        :param z: is the latent representation with shape `[batch_size, emb_channels, z_height, z_height]`
        """
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)
        recon_obsnorm = self.obsnorm_decoder(z, y)
        # Decode the image of shape `[batch_size, channels, height, width]`
        return self.decoder([self.dummy_actor() for _ in range(z.shape[0])], z, y), recon_obsnorm

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        if x is None:
            # Not coded for obsnorm yet
            posterior = self.random_sample(x, y)
        else:
            posterior = self.encode(x, y)
        z = posterior.sample()
        out = self.decode(z, y)
        return out, posterior


class ObsNormEncoder(nn.Module):
    def __init__(self,
                 obs_shape: int,
                 z_channels: int,
                 z_height: int = 4,
                 conditional: bool = False,
                 hid = 64,
                 ):
        super().__init__()
        self.z_channels = z_channels
        self.z_height = z_height
        self.conditional = conditional

        self.mean_enc = nn.Sequential(
            nn.Linear(obs_shape, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )
        self.std_enc = nn.Sequential(
            nn.Linear(obs_shape, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )
        self.z_enc = nn.Linear(2*hid, 2 * z_channels * z_height * z_height)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        mean = self.mean_enc(x['obs_rms.mean'])
        std = self.std_enc(x['obs_rms.logstd'])
        z = self.z_enc(torch.cat([mean, std], dim=-1))
        z = z.view(z.shape[0], 2 * self.z_channels, self.z_height, self.z_height)
        return z


class ObsNormDecoder(nn.Module):
    def __init__(self,
                 obs_shape: int,
                 z_channels: int,
                 z_height: int = 4,
                 conditional: bool = False,
                 hid = 64,
                 ):
        super().__init__()
        self.z_channels = z_channels
        self.z_height = z_height
        self.conditional = conditional
        inp_dim = z_channels * z_height * z_height

        if self.conditional:
            inp_dim += 2

        self.mean_dec = nn.Sequential(
            nn.Linear(inp_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, obs_shape)
        )
        self.std_dec = nn.Sequential(
            nn.Linear(inp_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, obs_shape)
        )
        
    def forward(self, z: torch.Tensor, y: torch.Tensor = None):
        z = z.view(z.shape[0], self.z_channels * self.z_height * self.z_height)
        if self.conditional:
            z = torch.cat([z, y], dim=-1)
        mean = self.mean_dec(z)
        std = self.std_dec(z)
        return {'obs_normalizer.obs_rms.mean': mean, 'obs_normalizer.obs_rms.logstd': std}


class ModelEncoder(nn.Module):
    def __init__(self, obs_shape, action_shape, emb_channels, z_channels, z_height=4,
                 regress_to_measure=False, measure_dim=2, conditional=False, fc_hid=64):
        super().__init__()

        self.conditional = conditional
        dummy_actor = Actor(obs_shape=obs_shape,
                            action_shape=action_shape,
                            normalize_obs=True,
                            normalize_returns=False,
                            deterministic=True)
        # TODO: maybe register the obs normalizer state dict as params in the policy state dict so that
        # TODO: we don't need to do this hack
        actor_state_dict = dummy_actor.state_dict()
        del actor_state_dict['obs_normalizer.obs_rms.count']

        self.channels = [1, 32, 64, 64, 128, 256, 256, 512, 512, 512]
        self.kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 2]
        self.strides = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.paddings = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

        self.max_pool_kernel_sizes = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        self.max_pool_strides = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
        self.z_channels = z_channels
        self.emb_channels = emb_channels
        self.z_height = z_height
        self.fc_hid = fc_hid
        self.regress_to_measure = regress_to_measure
        assert not (self.regress_to_measure and self.conditional), "Cannot regress to measure and be conditional on measure at the same time"
        self.measure_dim = measure_dim

        self.cnns = {}
        total_op_shape = 0
        self.list_of_weight_names = []
        self.list_of_real_weight_names = []
        for name, param in actor_state_dict.items():

            self.list_of_real_weight_names.append(name)
            key_name = "_".join(name.split('.'))
            self.list_of_weight_names.append(key_name)

            if 'weight' in name:
                shape = (1, 1,) + tuple(param.data.shape)
                self.cnns[key_name], op_shape = self._create_cnn_backbone(shape)

            elif 'bias' in name or 'obs_normalizer' in name:
                shape = (1, 1,) + tuple(param.data.shape) + (1,)
                self.cnns[key_name], op_shape = self._create_fc_backbone(shape, num_hidden=self.fc_hid)

            else:
                shape = (1,) + tuple(param.data.shape) + (1,)
                self.cnns[key_name], op_shape = self._create_fc_backbone(shape, num_hidden=self.fc_hid)

            total_op_shape += np.prod(op_shape)

        self.cnns = nn.ModuleDict(self.cnns)
        # self.obsnorm_encoder = ObsNormEncoder(obs_shape=obs_shape,
        #                         z_channels=z_channels,
        #                         z_height=z_height,
        #                         conditional=conditional,
        #                         hid = 64,
        #                         )
        # total_op_shape += 2 * z_channels * z_height * z_height

        if self.conditional:
            total_op_shape += 2

        self.out = nn.Linear(total_op_shape, 2 * self.z_channels * self.z_height * self.z_height)
        if self.regress_to_measure:
            self.measure_out = nn.Linear(2 * self.z_channels * self.z_height * self.z_height, self.measure_dim)

    # create a cnn backbone to extract features from tensor of shape (batch_size, *shape)
    def _create_cnn_backbone(self, shape, leaky_relu=False):

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            # shape of input: (batch, input_channel, height, width)
            input_channel = self.channels[i]
            output_channel = self.channels[i + 1]

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
            dim1 = (input_shape[2] - kernel_size + 2 * padding) // stride + 1
            dim2 = (input_shape[3] - kernel_size + 2 * padding) // stride + 1
            dim1 = (dim1 - pool_size) // pool_stride + 1
            dim2 = (dim2 - pool_size) // pool_stride + 1
            return (1, channel, int(dim1), int(dim2))

        for i in range(len(self.channels) - 2):
            channel = self.channels[i + 1]
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

    def _create_fc_backbone(self, shape, num_hidden):
        fc = nn.Sequential()
        fc.add_module('fc1', nn.Linear(shape[2], num_hidden))
        fc.add_module('relu1', nn.ReLU(inplace=True))
        fc.add_module('fc2', nn.Linear(num_hidden, num_hidden))
        fc.add_module('relu2', nn.ReLU(inplace=True))
        fc.add_module('fc3', nn.Linear(num_hidden, num_hidden))
        fc.add_module('relu3', nn.ReLU(inplace=True))
        return fc, (1, 1, num_hidden, 1)

    def forward(self, x, y = None, get_intermediate_features=False):
        outs = []
        features = []
        for k in range(len(self.list_of_weight_names)):
            name = self.list_of_weight_names[k]
            real_name = self.list_of_real_weight_names[k]
            if 'weight' in name:
                out = self.cnns[name](x[real_name].unsqueeze(1))
                if get_intermediate_features:
                    features.append(out)
                out = out.view(out.size(0), -1)
                outs.append(out)
            elif 'bias' in name or 'obs_normalizer' in name:
                out = self.cnns[name](x[real_name].unsqueeze(1))
                if get_intermediate_features:
                    features.append(out[:, :, :, None])
                out = out.view(out.size(0), -1)
                outs.append(out)
            else:
                out = self.cnns[name](x[real_name].unsqueeze(1))
                if get_intermediate_features:
                    features.append(out)
                out = out.view(out.size(0), -1)
                outs.append(out)


        # obs_norm_enc = self.obsnorm_encoder(x2)
        # outs.append(obs_norm_enc.flatten(1))
        x = torch.cat(outs, dim=1)
        if self.conditional:
            x = torch.cat([x, y], dim=1)
        x = self.out(x)
        if get_intermediate_features:
            features.append(x[:, :, None, None])
            # TODO: @Shank should we also append the outputs of self.measure_out? I think
            # TODO: since self.out() is the 2nd to last layer, this should be enough?,
            # TODO: @Sumeet yea lets add it just in case
            # TODO: Clean this
            if self.regress_to_measure:
                features.append(self.measure_out(x)[:, :, None, None])
            return [features[-1]]
        if not self.regress_to_measure:
            return x.reshape(-1, 2 * self.z_channels, self.z_height, self.z_height)
        else:
            return self.measure_out(x), x

    def to(self, device):
        # self.dummy_actor.to(device)
        for cnn in self.cnns.values():
            cnn.to(device)
        self.out.to(device)
        if self.regress_to_measure:
            self.measure_out.to(device)
        return self
