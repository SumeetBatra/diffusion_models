import pickle
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union, Optional
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from RL.actor_critic import Actor
# from ribs.archives._elite import EliteBatch
from tqdm import tqdm


def readonly(arr):
    """Sets an array to be readonly."""
    arr.flags.writeable = False
    return arr


class ShapedEliteDataset(Dataset):
    def __init__(self, archive_dfs: list[DataFrame],
                 obs_dim: int,
                 action_shape: Union[tuple, np.ndarray],
                 device: str,
                 normalize_obs: bool = False,
                 is_eval: bool = False,
                 inp_coef: float = 0.25,
                 eval_batch_size: Optional[int] = 8):
        archive_df = pandas.concat(archive_dfs)

        self.obs_dim = obs_dim
        self.action_shape = action_shape
        self.device = device
        self.is_eval = is_eval
        self.inp_coef = inp_coef if normalize_obs else 1

        self.measures_list = archive_df.filter(regex='measure*').to_numpy()
        self.metadata = archive_df.filter(regex='metadata*').to_numpy()
        self.normalize_obs = normalize_obs

        elites_list = archive_df.filter(regex='solution*').to_numpy()

        if self.is_eval:
            indices = np.random.choice(len(elites_list), eval_batch_size, replace=False)
            elites_list = elites_list[indices]
            self.measures_list = self.measures_list[indices]
            self.metadata = self.metadata[indices]

        self.weight_dicts_list, self.obsnorm_list = self._params_to_weight_dicts(elites_list)

    def __len__(self):
        return len(self.weight_dicts_list)

    def __getitem__(self, item):
        weights_dict, measures = self.weight_dicts_list[item], self.measures_list[item]
        obsnorm = self.obsnorm_list[item]
        return weights_dict, measures, obsnorm

    def _params_to_weight_dicts(self, elites_list):
        weight_dicts = []
        obsnorms = []
        for i, params in tqdm(enumerate(elites_list)):
            weights_dict = Actor(self.obs_dim, self.action_shape, self.normalize_obs, True).to(self.device).get_deserialized_weights(params)
            obs_normalizer = self.metadata[i][0]['obs_normalizer']
            if self.normalize_obs:
                weights_dict = self._integrate_obs_normalizer(weights_dict, obs_normalizer, self.inp_coef)
            weight_dicts.append(weights_dict)
            obsnorms.append(obs_normalizer.state_dict())
        return weight_dicts, obsnorms

    @staticmethod
    def _integrate_obs_normalizer(weights_dict, obs_normalizer, inp_coef):
        w_in = weights_dict['actor_mean.0.weight']
        b_in = weights_dict['actor_mean.0.bias']
        mean, var = obs_normalizer.obs_rms.mean, obs_normalizer.obs_rms.var

        w_new = inp_coef * (w_in / torch.sqrt(var + 1e-8))
        b_new = inp_coef * (b_in - (mean / torch.sqrt(var + 1e-8)) @ w_in.T)
        weights_dict['actor_mean.0.weight'] = w_new
        weights_dict['actor_mean.0.bias'] = b_new
        return weights_dict




if __name__ == '__main__':
    archive_df_path = '/home/sumeet/QDPPO/experiments/ppga_halfcheetah_100x100_no_obs_norm/1111/checkpoints/cp_00002000/archive_df_00002000.pkl'
    with open(archive_df_path, 'rb') as f:
        archive_df = pickle.load(f)

    mlp_shape = (128, 128, 6)
    obs_dim, action_shape = 18, np.array([6])

    dummy_agent = Actor(obs_shape=18, action_shape=np.array([6]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = ShapedEliteDataset([archive_df], obs_dim=obs_dim, action_shape=action_shape, device=device, normalize_obs=True)




