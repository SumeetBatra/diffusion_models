import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
from torch.distributions import MultivariateNormal, Categorical
from RL.normalize_obs import ReturnNormalizer, ObsNormalizer


class StochasticPolicy(ABC, nn.Module):
    def __init__(self, normalize_obs=False, obs_shape=None, normalize_returns=False):
        super().__init__()
        self.layers: nn.Sequential

        if normalize_obs:
            assert obs_shape is not None, 'Normalize obs is enabled but no obs_shape was given'
        self.obs_normalizer = ObsNormalizer(obs_shape) if normalize_obs else None
        self.return_normalizer = ReturnNormalizer() if normalize_returns else None

    @abstractmethod
    def forward(self, obs):
        pass

    @abstractmethod
    def get_action(self, obs, action=None):
        ...

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    # From DQDRL paper https://arxiv.org/pdf/2202.03666.pdf
    def serialize(self):
        '''
        Returns a 1D numpy array view of the entire policy.
        '''
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array: np.ndarray):
        '''
        Update the weights of this policy with the weights from the 1D
        array of parameters
        '''
        """Loads parameters from 1D array."""
        array = np.copy(array)
        arr_idx = 0
        for param in self.parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            param.data = torch.from_numpy(block).float()
        return self

    def get_deserialized_weights(self, array: np.ndarray):
        '''
        Returns a copy of the weights of this policy with the weights from the 1D
        array of parameters
        '''
        """Loads parameters from 1D array."""
        array = np.copy(array)
        arr_idx = 0
        weights = {}
        for name, param in self.named_parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            # param.data = torch.from_numpy(block).float()
            weights[name] = torch.from_numpy(block).to(torch.float32).to(param.device)
        return weights

    def gradient(self):
        '''Returns 1D numpy array view of the gradients of this actor'''
        return np.concatenate(
            [p.grad.cpu().detach().numpy().ravel() for p in self.parameters()]
        )
