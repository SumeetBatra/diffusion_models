import pickle
import pandas
import torch
import os
import numpy as np

from typing import List, Union, Optional
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from RL.actor_critic import Actor
# from ribs.archives._elite import EliteBatch
from tqdm import tqdm
from utils.normalize import ObsNormalizer
from utils.tensor_dict import TensorDict, cat_tensordicts


def readonly(arr):
    """Sets an array to be readonly."""
    arr.flags.writeable = False
    return arr


class WeightNormalizer:
    def __init__(self, means_dict: TensorDict, std_dict: TensorDict):
        self.means_dict = means_dict
        self.std_dict = std_dict

    def normalize(self, data: TensorDict):
        for name, param in data.items():
            data[name] = (param - self.means_dict[name]) / (self.std_dict[name] + 1e-8)

        return data

    def denormalize(self, data: TensorDict):
        for name, param in data.items():
            data[name] = param * (self.std_dict[name] + 1e-8) + self.means_dict[name]

        return data

    def save(self, save_path: str):
        data = {
            'means_dict': dict(self.means_dict),
            'std_dict': dict(self.std_dict)
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, save_path: str):
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        self.means_dict = TensorDict(data['means_dict'])
        self.std_dict = TensorDict(data['std_dict'])


class ShapedEliteDataset(Dataset):
    def __init__(self,
                 archive_dfs: list[DataFrame],
                 obs_dim: int,
                 action_shape: Union[tuple, np.ndarray],
                 device: str,
                 is_eval: bool = False,
                 eval_batch_size: Optional[int] = 8,
                 center_data: bool = False,
                 cut_out: bool = False,
                 weight_normalizer: Optional[WeightNormalizer] = None):
        archive_df = pandas.concat(archive_dfs)

        self.obs_dim = obs_dim
        self.action_shape = action_shape
        self.device = device
        self.is_eval = is_eval
        self.cut_out = cut_out

        self.measures_list = archive_df.filter(regex='measure*').to_numpy()
        self.metadata = archive_df.filter(regex='metadata*').to_numpy()
        self.objective_list = archive_df['objective'].to_numpy()

        elites_list = archive_df.filter(regex='solution*').to_numpy()

        if cut_out:
            indices_to_cut = np.argwhere((self.measures_list[:,0] > 0.5) * (self.measures_list[:,1] > 0.5) * (self.measures_list[:,0] < 0.6) * (self.measures_list[:,1] < 0.6))
            elites_list = np.delete(elites_list, indices_to_cut, axis=0)
            self.measures_list = np.delete(self.measures_list, indices_to_cut, axis=0)
            self.metadata = np.delete(self.metadata, indices_to_cut, axis=0)
            self.objective_list = np.delete(self.objective_list, indices_to_cut, axis=0)


        if self.is_eval:
            # indices shall be eval_batch_size number of indices spaced out (by objective) evenly across the elites_list
            indices = np.linspace(0, len(elites_list) - 1, eval_batch_size, dtype=int)
            indices = np.argsort(archive_df['objective'].to_numpy())[indices]
            self.indices = indices
            elites_list = elites_list[indices]
            self.measures_list = self.measures_list[indices]
            self.metadata = self.metadata[indices]
            self.objective_list = self.objective_list[indices]

        self._size = len(elites_list)

        weight_dicts_list = self._params_to_weight_dicts(elites_list)
        self.weights_dict = cat_tensordicts(weight_dicts_list)

        # per-layer mean and std-dev stats for centering / de-centering the data
        if weight_normalizer is None:
            weight_mean_dict = TensorDict({
                key: self.weights_dict[key].mean(0).to(self.device) for key in self.weights_dict.keys()
            })

            weight_std_dict = TensorDict({
                key: self.weights_dict[key].std(0).to(self.device) for key in self.weights_dict.keys()
            })
            weight_normalizer = WeightNormalizer(means_dict=weight_mean_dict, std_dict=weight_std_dict)

        self.normalizer = weight_normalizer

        # zero center the data with unit variance
        if center_data:
            self.weights_dict = self.normalizer.normalize(self.weights_dict)

    def __len__(self):
        return self._size

    def __getitem__(self, item):
        weights_dict, measures = self.weights_dict[item], self.measures_list[item]
        return weights_dict, measures

    def _params_to_weight_dicts(self, elites_list):
        weight_dicts = []
        for i, params in tqdm(enumerate(elites_list)):
            agent = Actor(self.obs_dim, self.action_shape, True, False)
            normalize_obs = self.metadata[i][0]['obs_normalizer']
            if isinstance(normalize_obs, dict):
                obs_normalizer = ObsNormalizer(self.obs_dim).to(self.device)
                obs_normalizer.load_state_dict(normalize_obs)
                agent.obs_normalizer = obs_normalizer
            else:
                agent.obs_normalizer = normalize_obs

            weights_dict = TensorDict(agent.deserialize(params).to(self.device).state_dict())
            weights_dict['obs_normalizer.obs_rms.std'] = torch.sqrt(weights_dict['obs_normalizer.obs_rms.var'] + 1e-8)
            weights_dict['obs_normalizer.obs_rms.logstd'] = torch.log(weights_dict['obs_normalizer.obs_rms.std'])
            weight_dicts.append(weights_dict)
        return weight_dicts


class LangShapedEliteDataset(ShapedEliteDataset):

    def __init__(self, *args, text_labels: List[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.text_labels = text_labels
        if self.is_eval:
            self.text_labels = [self.text_labels[i] for i in self.indices]

    def __getitem__(self, item):
        weights_dict, measures = super().__getitem__(item)
        return weights_dict, (measures, self.text_labels[item])


if __name__ == '__main__':
    archive_df_path = '/home/sumeet/QDPPO/experiments/ppga_halfcheetah_100x100_no_obs_norm/1111/checkpoints/cp_00002000/archive_df_00002000.pkl'
    with open(archive_df_path, 'rb') as f:
        archive_df = pickle.load(f)

    mlp_shape = (128, 128, 6)
    obs_dim, action_shape = 18, np.array([6])

    dummy_agent = Actor(obs_shape=18, action_shape=np.array([6]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = ShapedEliteDataset([archive_df], obs_dim=obs_dim, action_shape=action_shape, device=device, normalize_obs=True)
