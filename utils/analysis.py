import torch
import torch.nn as nn
import numpy as np
import pickle
import os

from attrdict import AttrDict
from autoencoders.policy.hypernet import HypernetAutoEncoder as VAE
from collections import OrderedDict
from ribs.archives import GridArchive
from utils.brax_utils import rollout_many_agents
from utils.archive_utils import archive_df_to_archive, reevaluate_ppga_archive, save_heatmap
from utils.brax_utils import shared_params
from envs.brax_custom import reward_offset


def evaluate_vae_subsample(env_name: str, archive_df = None, model = None, N: int = 100, image_path: str = None, suffix: str = None, ignore_first: bool = False):
    '''Randomly sample N elites from the archive. Evaluate the original elites and the reconstructed elites
    from the VAE. Compare the performance using a subsampled QD-Score. Compare the behavior accuracy using the l2 norm
    :param env_name: Name of the environment ex walker2d
    :param model_path: Path to the VAE model
    :param archive_df_path: Path to the archive df(s) used to train the VAE
    :param N: number of samples from the archive to evaluate. If N is set to -1, we will evaluate the entire archive, but
    be warned -- this is really expensive, especially for larger archives!
    :param image_path: Path to save the heatmap images
    :param suffix: Suffix to append to the heatmap image name
    :param ignore_first: If True, we will not evaluate the original archive. This is useful if you want to compare the performance
    '''

    if type(model) == str:
        vae = VAE(emb_channels=8, z_channels=4)
        vae.load_state_dict(torch.load(model))
    else:
        vae = model


    if type(archive_df) == str:
        with open(archive_df, 'rb') as f:
            archive_df = pickle.load(f)
    else:
        archive_df = archive_df

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
    if not ignore_first:
        print('Re-evaluated Original Archive')
        original_reevaluated_archive = reevaluate_ppga_archive(env_cfg,
                                normalize_obs,
                                normalize_returns,
                                original_archive)
        original_results = {
                'Coverage': original_reevaluated_archive.stats.coverage,
                'Max_fitness': original_reevaluated_archive.stats.obj_max,
                'Avg_Fitness': original_reevaluated_archive.stats.obj_mean,
                'QD_Score': original_reevaluated_archive.offset_qd_score
        }

    print('Re-evaluated Reconstructed Archive')
    reconstructed_evaluated_archive = reevaluate_ppga_archive(env_cfg,
                            normalize_obs,
                            normalize_returns,
                            original_archive,
                            reconstructed_agents=True,
                            vae=vae)
    reconstructed_results = {
            'Coverage': reconstructed_evaluated_archive.stats.coverage,
            'Max_fitness': reconstructed_evaluated_archive.stats.obj_max,
            'Avg_Fitness': reconstructed_evaluated_archive.stats.obj_mean,
            'QD_Score': reconstructed_evaluated_archive.offset_qd_score
    }
    results = {
        'Original': original_results if not ignore_first else None,
        'Reconstructed': reconstructed_results,
    }


    if image_path is not None:
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not ignore_first:
            orig_image_array = save_heatmap(original_reevaluated_archive, os.path.join(image_path, f"original_archive_{suffix}.png"))
        recon_image_array = save_heatmap(reconstructed_evaluated_archive, os.path.join(image_path, f"reconstructed_archive_{suffix}.png"))

    image_results = {
        'Original': orig_image_array if not ignore_first else None,
        'Reconstructed': recon_image_array,
    }
    return results, image_results



if __name__ == '__main__':
    archive_df_path = '/home/sumeet/QDPPO/experiments/ppga_halfcheetah_adaptive_stddev_no_obs_norm/1111/checkpoints/' \
                      'cp_00001990/archive_df_00001990.pkl'

    model_path = 'checkpoints/autoencoder.pt'

    env_name = 'halfcheetah'

    evaluate_vae_subsample(env_name, archive_df_path, model_path)