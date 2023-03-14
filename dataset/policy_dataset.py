import pickle
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from RL.actor_critic import Actor


def preprocess_model(model: nn.Module, mlp_shape: Union[list, tuple]):
    '''Performs the same ops as ElitesDataset._preprocess_data(), except on a single model'''
    largest_layer_size = max(mlp_shape)
    params = model.serialize()
    padded_policy = []
    arr_idx = 0
    for name, param in model.named_parameters():
        if name == 'actor_logstd':
            arr_idx += mlp_shape[-1]
            continue
        if 'weight' in name:
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = params[arr_idx: arr_idx + length]
            block = torch.from_numpy(np.reshape(block, shape))
        else:
            # bias term
            orig_shape = length = param.data.shape[0]
            block = params[arr_idx: arr_idx + length]
            block = torch.from_numpy(block).repeat(orig_shape).reshape(orig_shape, orig_shape)
            shape = tuple(block.shape)
            # pad all sides of the weight matrix so that the final shape is (largest_layer_size x largest_layer_size)
        padding = ((largest_layer_size - shape[1]) // 2, (largest_layer_size - shape[1]) // 2,
                   (largest_layer_size - shape[0]) // 2, (largest_layer_size - shape[0]) // 2)
        block = F.pad(block, pad=padding)
        padded_policy.append(block)
        arr_idx += length
    padded_policy = torch.stack(padded_policy).unsqueeze(dim=0).type(
        torch.float32)  # this makes it (1 x depth x largest_layer_size x largest_layer_size)
    return padded_policy


def postprocess_model(model_in: nn.Module, padded_params: torch.Tensor, mlp_shape: Union[list, tuple]):
    '''Reconstruct the original policy given the padded policy tensor'''
    largest_layer_size = max(mlp_shape)
    i = 0
    all_params = []
    for name, param in model_in.named_parameters():
        if name == 'actor_logstd':
            all_params.append(np.zeros(mlp_shape[-1]))
            continue
        if 'weight' in name:
            shape = tuple(param.data.shape)
            padding = ((largest_layer_size - shape[0]) // 2, (largest_layer_size - shape[0]) // 2,
                       (largest_layer_size - shape[1]) // 2, (largest_layer_size - shape[1]) // 2)
            layer = padded_params[0][i]

            actual_params = layer[padding[0]:padding[0] + param.data.shape[0],
                            padding[2]:padding[2] + param.data.shape[1]]
        else:
            # bias term
            shape = param.data.shape[0]
            padding = ((largest_layer_size - shape) // 2, (largest_layer_size - shape) // 2,
                       (largest_layer_size - shape) // 2, (largest_layer_size - shape) // 2)
            layer = padded_params[0][i]

            actual_params = layer[0][padding[0]:padding[0] + shape]

        all_params.append(actual_params.detach().cpu().numpy().flatten())
        i += 1
    all_params = np.concatenate(all_params).flatten()
    return model_in.deserialize(all_params)


class ElitesDataset(Dataset):
    def __init__(self, archive_dfs: list[DataFrame], mlp_shape: Union[tuple, list], dummy_model: nn.Module):
        self.elites_list = []
        self.dummy_model = dummy_model
        self._preprocess_data(archive_dfs, mlp_shape)

    def __len__(self):
        return len(self.elites_list)

    def __getitem__(self, item):
        return self.elites_list[item]

    def _preprocess_data(self, archive_dfs, mlp_shape):
        '''
        Given a mlp of depth D and largest hidden layer of size L, we will pad all the network weights
        to produce tensors of size (L x L x D)
        '''
        archive_df = pandas.concat(archive_dfs)
        params_batch = archive_df.filter(regex='solution*').to_numpy()
        largest_layer_size = max(mlp_shape)

        padded_elites = []
        for params in params_batch:
            padded_policy = []
            arr_idx = 0
            for name, param in self.dummy_model.named_parameters():
                if name == 'actor_logstd':
                    arr_idx += mlp_shape[-1]
                    continue
                if 'weight' in name:
                    shape = tuple(param.data.shape)
                    length = np.product(shape)
                    block = params[arr_idx: arr_idx + length]
                    block = torch.from_numpy(np.reshape(block, shape))
                else:
                    # bias term
                    orig_shape = length = param.data.shape[0]
                    block = params[arr_idx: arr_idx + length]
                    block = torch.from_numpy(block).repeat(orig_shape).reshape(orig_shape, orig_shape)
                    shape = tuple(block.shape)
                # pad all sides of the weight matrix so that the final shape is (largest_layer_size x largest_layer_size)
                padding = ((largest_layer_size - shape[1]) // 2, (largest_layer_size - shape[1]) // 2,
                           (largest_layer_size - shape[0]) // 2, (largest_layer_size - shape[0]) // 2)
                block = F.pad(block, pad=padding)
                padded_policy.append(block)
                arr_idx += length
            padded_policy = torch.stack(padded_policy).unsqueeze(dim=0).type(torch.float32)  # this makes it (1 x depth x largest_layer_size x largest_layer_size)
            padded_elites.append(padded_policy)
        self.elites_list = padded_elites


if __name__ == '__main__':
    archive_df_path = '/home/sumeet/QDPPO/experiments/paper_qdppo_halfcheetah/1111/checkpoints/cp_00002000/archive_00002000.pkl'
    with open(archive_df_path, 'rb') as f:
        archive_df = pickle.load(f)

    mlp_shape = (128, 128, 6)

    dummy_agent = Actor(obs_shape=18, action_shape=np.array([6]))

    ds = ElitesDataset(archive_df, mlp_shape, dummy_agent)




