import torch
import torch.nn as nn
import numpy as np
import pickle

from attrdict import AttrDict
from autoencoders.policy.hypernet import HypernetAutoEncoder as VAE
from collections import OrderedDict
from ribs.archives import GridArchive
from utils.brax_utils import rollout_many_agents
from utils.archive_utils import archive_df_to_archive, reevaluate_ppga_archive
from utils.brax_utils import shared_params
from envs.brax_custom import reward_offset


def evaluate_vae_subsample(env_name: str, model_path: str, archive_df_path: str, N: int = 100):
    '''Randomly sample N elites from the archive. Evaluate the original elites and the reconstructed elites
    from the VAE. Compare the performance using a subsampled QD-Score. Compare the behavior accuracy using the l2 norm
    :param env_name: Name of the environment ex walker2d
    :param model_path: Path to the VAE model
    :param archive_df_path: Path to the archive df(s) used to train the VAE
    :param N: number of samples from the archive to evaluate. If N is set to -1, we will evaluate the entire archive, but
    be warned -- this is really expensive, especially for larger archives!
    '''

    vae = VAE(emb_channels=8, z_channels=4)
    vae.load_state_dict(torch.load(model_path))

    with open(archive_df_path, 'rb') as f:
        archive_df = pickle.load(f)

    env_cfg = AttrDict(shared_params[env_name]['env_cfg'])
    env_cfg.seed = 1111

    if N != -1:
        archive_df = archive_df.sample(N)

    soln_dim = archive_df.filter(regex='solution*').to_numpy().shape[1]
    archive_dims = [env_cfg['grid_size']] * env_cfg['num_dims']
    ranges = [(0.0, 1.0)] * env_cfg['num_dims']
    original_archive = archive_df_to_archive(archive_df,
                                             solution_dim=soln_dim,
                                             dims=archive_dims,
                                             ranges=ranges,
                                             seed=env_cfg.seed,
                                             qd_offset=reward_offset[env_name])

    normalize_obs, normalize_returns = True, True
    print('Re-evaluated Original Archive')
    reevaluate_ppga_archive(env_cfg,
                            normalize_obs,
                            normalize_returns,
                            original_archive)

    print('Re-evaluated Reconstructed Archive')
    reevaluate_ppga_archive(env_cfg,
                            normalize_obs,
                            normalize_returns,
                            original_archive,
                            reconstructed_agents=True,
                            vae=vae)




if __name__ == '__main__':
    archive_df_path = '/home/sumeet/QDPPO/experiments/ppga_halfcheetah_adaptive_stddev_no_obs_norm/1111/checkpoints/' \
                      'cp_00001990/archive_df_00001990.pkl'

    model_path = 'checkpoints/autoencoder.pt'

    env_name = 'halfcheetah'

    evaluate_vae_subsample(env_name, model_path, archive_df_path)