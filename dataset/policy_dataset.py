import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from RL.actor_critic import Actor

from ribs.archives._elite import EliteBatch
from attrdict import AttrDict
from tqdm import tqdm

def readonly(arr):
    """Sets an array to be readonly."""
    arr.flags.writeable = False
    return arr


class ElitesDataset(Dataset):
    def __init__(self, archive_df: DataFrame, mlp_shape: Union[tuple, list], dummy_model: nn.Module):
        self.elites_list = []
        self.dummy_model = dummy_model
        self._preprocess_data(archive_df, mlp_shape)

    def __len__(self):
        return len(self.elites_list)

    def __getitem__(self, item):
        return self.elites_list[item]

    def _preprocess_data(self, archive_df, mlp_shape):
        '''
        Given a mlp of depth D and largest hidden layer of size L, we will pad all the network weights
        to produce tensors of size (L x L x D)
        '''
        params_batch = archive_df.filter(regex='solution*').to_numpy()
        largest_layer_size = max(mlp_shape)

        padded_elites = []
        for params in params_batch:
            padded_policy = []
            arr_idx = 0
            for name, param in self.dummy_model.named_parameters():
                if name == 'actor_logstd':
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


class ShapedEliteDataset(Dataset):
    def __init__(self, actor_cfg, elites, device):
        self.elites = elites
        self.actor_cfg = actor_cfg
        self.device = device
        self.elite_list = []
        self._get_all_elite()


    def __len__(self):
        return len(self.elite_list)
    
    def __getitem__(self, idx):
        return self.elite_list[idx]

    
    def _get_all_elite(self):
        self.elite_list = []
        for i in tqdm(range(len(self.elites.solution_batch))):
            params = self.elites.solution_batch[i].flatten()
            weights_dict = Actor(self.actor_cfg, obs_shape=self.actor_cfg.obs_shape[0], action_shape=self.actor_cfg.action_shape) \
                .to(self.device).get_deserialized_weights(params)
            
            elite_measure = self.elites.measures_batch[i]

            # convert elite_measure to float 32
            elite_measure = torch.tensor(elite_measure, dtype=torch.float32)

            elite_measure = elite_measure.to(self.device)

            if torch.isinf(elite_measure).any().item() or torch.max(elite_measure) > 1 or torch.min(elite_measure) < 0:
                continue
            else:
                self.elite_list.append((weights_dict, elite_measure))

archive_path = '/home/shashank/research/qd/arxiv/ppga_halfcheetah_100x100/1111/checkpoints/cp_00002000/archive_00002000.pkl'
scheduler_path = '/home/shashank/research/qd/arxiv/ppga_halfcheetah_100x100/1111/checkpoints/cp_00002000/scheduler_00002000.pkl'

with open(archive_path, 'rb') as f:
    archive_df = pickle.load(f)
with open(scheduler_path, 'rb') as f:
    scheduler = pickle.load(f)

total_number_of_elites = scheduler.archive._num_occupied
all_indices = np.arange(total_number_of_elites)
np.random.shuffle(all_indices)
train_indices = all_indices[:int(0.9*total_number_of_elites)]
test_indices = all_indices[int(0.9*total_number_of_elites):]

train_elites = EliteBatch(
            readonly(scheduler.archive._solution_arr[train_indices]),
            readonly(scheduler.archive._objective_arr[train_indices]),
            readonly(scheduler.archive._measures_arr[train_indices]),
            readonly(train_indices),
            readonly(scheduler.archive._metadata_arr[train_indices]),
        )

test_elites = EliteBatch(
            readonly(scheduler.archive._solution_arr[test_indices]),
            readonly(scheduler.archive._objective_arr[test_indices]),
            readonly(scheduler.archive._measures_arr[test_indices]),
            readonly(test_indices),
            readonly(scheduler.archive._metadata_arr[test_indices]),
        )    

device = torch.device('cuda')
env_name = 'halfcheetah'
seed = 1111
normalize_obs = True
normalize_rewards = True
# non-configurable params
obs_shapes = {
    'humanoid': (227,),
    'ant': (87,),
    'halfcheetah': (18,),
    'walker2d': (17,)
}
action_shapes = {
    'humanoid': (17,),
    'ant': (8,),
    'halfcheetah': (6,),
    'walker2d': (6,)
}

# define the final config objects
actor_cfg = AttrDict({
        'obs_shape': obs_shapes[env_name],
        'action_shape': action_shapes[env_name],
        'normalize_obs': normalize_obs,
        'normalize_rewards': normalize_rewards,
})
env_cfg = AttrDict({
        'env_name': env_name,
        'env_batch_size': None,
        'num_dims': 2 if not 'ant' in env_name else 4,
        'envs_per_model': 1,
        'seed': seed,
        'num_envs': 1,
})

batch_size = 32

e_dataset_train = ShapedEliteDataset(actor_cfg, train_elites, device)
e_data_loader_train = DataLoader(e_dataset_train, batch_size=batch_size, shuffle=True)

e_dataset_test = ShapedEliteDataset(actor_cfg, test_elites, device)
e_data_loader_test = DataLoader(e_dataset_test, batch_size=batch_size, shuffle=True)

# e_data_loader_train = DataLoader(e_dataset_test, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    archive_df_path = '/home/sumeet/QDPPO/experiments/paper_qdppo_halfcheetah/1111/checkpoints/cp_00002000/archive_00002000.pkl'
    with open(archive_df_path, 'rb') as f:
        archive_df = pickle.load(f)

    mlp_shape = (128, 128, 6)

    dummy_agent = Actor(obs_shape=18, action_shape=np.array([6]))

    ds = ElitesDataset(archive_df, mlp_shape, dummy_agent)




