from models.cond_unet import ConditionalUNet
from autoencoders.policy.hypernet import HypernetAutoEncoder as AutoEncoder
import torch
from utils.brax_utils import shared_params
import numpy as np
import glob
import pickle
from diffusion.gaussian_diffusion import cosine_beta_schedule
from diffusion.latent_diffusion import LatentDiffusion
from diffusion.ddim import DDIMSampler
from algorithm.train_autoencoder import shaped_elites_dataset_factory
import json
from attrdict import AttrDict
from envs.brax_custom import reward_offset
from utils.archive_utils import archive_df_to_archive,reevaluate_ppga_archive
from ribs.archives import GridArchive
from envs.brax_custom.brax_env import make_vec_env_brax
from RL.actor_critic import Actor
from utils.archive_utils import reconstruct_agents_from_ldm, evaluate
import torch.nn as nn
from models.vectorized import VectorizedActor
import os
import copy
import random
import matplotlib.pyplot as plt
import pandas as pd

experiments_dict_old = {
    "results": [
        {
            "env": "humanoid",
            "experiments_dict_list" : [
                {
                    "centering": True,
                    "ghn_size": 8,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "humanoid_diffusion_model_20230504-041644_111"
                        },
                        {
                            "seed": 222,
                            "name": "humanoid_diffusion_model_20230504-131631_222"
                        },
                        {
                            "seed": 333,
                            "name": "humanoid_diffusion_model_20230504-131635_333"
                        },
                        {
                            "seed": 444,
                            "name": "humanoid_diffusion_model_20230504-072944_444"
                        }
                    ],

                },

                {
                    "centering": False,
                    "ghn_size": 8,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "humanoid_diffusion_model_20230504-073156_111"
                        },
                        {
                            "seed": 222,
                            "name": "humanoid_diffusion_model_20230504-085106_222"
                        },
                        # {
                        #     "seed": 333,
                        #     "name": "humanoid_diffusion_model_20230504-090220_333"
                        # },
                        {
                            "seed": 444,
                            "name": "humanoid_diffusion_model_20230504-105735_444"
                        }
                    ],

                },

                {
                    "centering": True,
                    "ghn_size": 16,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "humanoid_diffusion_model_20230505-072152_111"
                        },
                        {
                            "seed": 222,
                            "name": "humanoid_diffusion_model_20230505-072152_222"
                        },
                        {
                            "seed": 333,
                            "name": "humanoid_diffusion_model_20230505-074108_333"
                        },
                        {
                            "seed": 444,
                            "name": "humanoid_diffusion_model_20230505-074238_444"
                        }
                    ],

                },

                {
                    "centering": False,
                    "ghn_size": 16,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "humanoid_diffusion_model_20230505-074434_111"
                        },
                        {
                            "seed": 222,
                            "name": "humanoid_diffusion_model_20230505-074447_222"
                        },
                        {
                            "seed": 333,
                            "name": "humanoid_diffusion_model_20230505-082740_333"
                        },
                        # {
                        #     "seed": 444,
                        #     "name": "humanoid_diffusion_model_20230504-072944_444"
                        # }
                    ],

                },

                {
                    "centering": True,
                    "ghn_size": 32,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "humanoid_diffusion_model_20230505-083900_111"
                        },
                        {
                            "seed": 222,
                            "name": "humanoid_diffusion_model_20230505-091645_222"
                        },
                        {
                            "seed": 333,
                            "name": "humanoid_diffusion_model_20230505-103651_333"
                        },
                        {
                            "seed": 444,
                            "name": "humanoid_diffusion_model_20230505-104603_444"
                        }
                    ],

                },

                {
                    "centering": False,
                    "ghn_size": 32,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "humanoid_diffusion_model_20230505-104603_111"
                        },
                        {
                            "seed": 222,
                            "name": "humanoid_diffusion_model_20230505-123637_222"
                        },
                        {
                            "seed": 333,
                            "name": "humanoid_diffusion_model_20230505-124223_333"
                        },
                        {
                            "seed": 444,
                            "name": "humanoid_diffusion_model_20230505-125828_444"
                        }
                    ],

                },

                # higher KL
                {
                    "centering": True,
                    "ghn_size": 16,
                    "kl": 1e-2,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "humanoid_diffusion_model_20230509-040115_111"
                        },
                        # {
                        #     "seed": 222,
                        #     "name": "humanoid_diffusion_model_20230505-072152_222"
                        # },
                        {
                            "seed": 333,
                            "name": "humanoid_diffusion_model_20230509-041637_333"
                        },
                        {
                            "seed": 444,
                            "name": "humanoid_diffusion_model_20230509-044508_444"
                        }
                    ],
                },
            ]
        },
        {
            "env": "walker2d",
            "experiments_dict_list" : [
                {
                    "centering": True,
                    "ghn_size": 8,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "walker2d_diffusion_model_20230503-232717_111"
                        },
                        {
                            "seed": 222,
                            "name": "walker2d_diffusion_model_20230503-233013_222"
                        },
                        {
                            "seed": 333,
                            "name": "walker2d_diffusion_model_20230504-002031_333"
                        },
                        {
                            "seed": 444,
                            "name": "walker2d_diffusion_model_20230504-010106_444"
                        }
                    ],

                },

                {
                    "centering": False,
                    "ghn_size": 8,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "walker2d_diffusion_model_20230504-014207_111"
                        },
                        {
                            "seed": 222,
                            "name": "walker2d_diffusion_model_20230504-041134_222"
                        },
                        {
                            "seed": 333,
                            "name": "walker2d_diffusion_model_20230504-020519_333"
                        },
                        {
                            "seed": 444,
                            "name": "walker2d_diffusion_model_20230504-041317_444"
                        }
                    ],

                },
            ]
        },
        {
            "env": "ant",
            "experiments_dict_list" : [
                {
                    "centering": True,
                    "ghn_size": 8,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "ant_diffusion_model_20230511-123612_111"
                        },
                        {
                            "seed": 222,
                            "name": "ant_diffusion_model_20230511-142040_222"
                        },
                        {
                            "seed": 333,
                            "name": "ant_diffusion_model_20230511-142040_333"
                        },
                        {
                            "seed": 444,
                            "name": "ant_diffusion_model_20230511-142040_444"
                        }
                    ],

                },

                {
                    "centering": False,
                    "ghn_size": 8,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "ant_diffusion_model_20230511-142100_111"
                        },
                        {
                            "seed": 222,
                            "name": "ant_diffusion_model_20230511-155513_222"
                        },
                        {
                            "seed": 333,
                            "name": "ant_diffusion_model_20230511-155513_333"
                        },
                        {
                            "seed": 444,
                            "name": "ant_diffusion_model_20230511-155513_444"
                        }
                    ],

                },
            ]
        },
        {
            "env": "halfcheetah",
            "experiments_dict_list" : [
                {
                    "centering": True,
                    "ghn_size": 8,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "halfcheetah_diffusion_model_20230507-112809_111"
                        },
                        {
                            "seed": 222,
                            "name": "halfcheetah_diffusion_model_20230507-112821_222"
                        },
                        {
                            "seed": 333,
                            "name": "halfcheetah_diffusion_model_20230507-112831_333"
                        },
                        {
                            "seed": 444,
                            "name": "halfcheetah_diffusion_model_20230507-142738_444"
                        }
                    ],

                },

                {
                    "centering": False,
                    "ghn_size": 8,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "halfcheetah_diffusion_model_20230507-150758_111"
                        },
                        {
                            "seed": 222,
                            "name": "halfcheetah_diffusion_model_20230507-151258_222"
                        },
                        {
                            "seed": 333,
                            "name": "halfcheetah_diffusion_model_20230507-151258_333"
                        },
                        {
                            "seed": 444,
                            "name": "halfcheetah_diffusion_model_20230507-151301_444"
                        }
                    ],

                },
            ]
        }        
    ]
}


experiments_dict = {
    "results": [
        {
            "env": "humanoid",
            "experiments_dict_list" : [
                {
                    "centering": True,
                    "ghn_size": 16,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "humanoid_diffusion_model_20230505-072152_111"
                        },
                        {
                            "seed": 222,
                            "name": "humanoid_diffusion_model_20230505-072152_222"
                        },
                        {
                            "seed": 333,
                            "name": "humanoid_diffusion_model_20230505-074108_333"
                        },
                        {
                            "seed": 444,
                            "name": "humanoid_diffusion_model_20230505-074238_444"
                        }
                    ],

                },

                {
                    "centering": False,
                    "ghn_size": 16,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "humanoid_diffusion_model_20230505-074434_111"
                        },
                        {
                            "seed": 222,
                            "name": "humanoid_diffusion_model_20230505-074447_222"
                        },
                        {
                            "seed": 333,
                            "name": "humanoid_diffusion_model_20230505-082740_333"
                        },
                        {
                            "seed": 444,
                            "name": "humanoid_diffusion_model_20230515-054355_444"
                        }
                    ],

                },       
            ]
        },
        {
            "env": "walker2d",
            "experiments_dict_list" : [
                {
                    "centering": False,
                    "ghn_size": 16,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "walker2d_diffusion_model_20230513-155129_111"
                        },
                        {
                            "seed": 222,
                            "name": "walker2d_diffusion_model_20230513-155141_222"
                        },
                        {
                            "seed": 333,
                            "name": "walker2d_diffusion_model_20230513-155155_333"
                        },
                        {
                            "seed": 444,
                            "name": "walker2d_diffusion_model_20230513-155207_444"
                        }
                    ],

                },
            ]
        },
        {
            "env": "ant",
            "experiments_dict_list" : [
                {
                    "centering": False,
                    "ghn_size": 16,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "ant_diffusion_model_20230513-155615_111"
                        },
                        {
                            "seed": 222,
                            "name": "ant_diffusion_model_20230513-155615_222"
                        },
                        {
                            "seed": 333,
                            "name": "ant_diffusion_model_20230513-155615_333"
                        },
                        {
                            "seed": 444,
                            "name": "ant_diffusion_model_20230513-160543_444"
                        }
                    ],

                },
                {
                    "centering": True,
                    "ghn_size": 16,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "ant_diffusion_model_20230514-141702_111"
                        },
                        {
                            "seed": 222,
                            "name": "ant_diffusion_model_20230514-134247_222"
                        },
                        {
                            "seed": 333,
                            "name": "ant_diffusion_model_20230514-134253_333"
                        },
                        {
                            "seed": 444,
                            "name": "ant_diffusion_model_20230514-134300_444"
                        }
                    ],
                }
            ]
        },
        {
            "env": "halfcheetah",
            "experiments_dict_list" : [
                {
                    "centering": False,
                    "ghn_size": 16,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "halfcheetah_diffusion_model_20230513-155047_111"
                        },
                        {
                            "seed": 222,
                            "name": "halfcheetah_diffusion_model_20230514-142656_222"
                        },
                        {
                            "seed": 333,
                            "name": "halfcheetah_diffusion_model_20230513-155059_333"
                        },
                        {
                            "seed": 444,
                            "name": "halfcheetah_diffusion_model_20230513-155106_444"
                        }
                    ],

                },
            ]
        }        
    ]
}

def make_cdf_plot(cfg, data: pd.DataFrame, ax: plt.axis, original: bool = False, standalone: bool = False, **kwargs):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if original:
        cfg.algorithm = "Original Archive CDF"
    else:
        cfg.algorithm = "Reconstructed Archive CDF"

    y_label = "Archive CDF"

    x = data['Objective'].to_numpy().flatten()
    y_avg = data.filter(regex='Mean').to_numpy().flatten()
    y_min = data.filter(regex='Min').to_numpy().flatten()
    y_max = data.filter(regex='Max').to_numpy().flatten()
    ax.plot(x, y_avg, linewidth=1.0, label=cfg.algorithm, **kwargs)
    ax.fill_between(x, y_min, y_max, alpha=0.2)
    ax.set_xlim(cfg.objective_range)
    ax.set_yticks(np.arange(0, 101, 25.0))
    ax.set_xlabel("Objective")
    if standalone:
        ax.set_ylabel(y_label)
        ax.set_title(cfg.title)
        ax.legend()

def compile_cdf(cfg, dataframes=None):
    num_cells = cfg.archive_resolution

    if not dataframes:
        df_dir = cfg.archive_dir
        filenames = next(os.walk(df_dir), (None, None, []))[2]  # [] if no file
        dataframes = []
        for f in filenames:
            full_path = os.path.join(df_dir, f)
            df = pd.read_pickle(full_path)
            dataframes.append(df)

    x = np.linspace(cfg.objective_range[0], cfg.objective_range[1], cfg.objective_resolution)
    all_y_vals = []
    for df in dataframes:
        y_vals = []
        df_cells = np.array(sorted(df['objective']))
        for x_val in x:
            count = len(df_cells[df_cells > x_val])
            percentage = (count / num_cells) * 100.0
            y_vals.append(percentage)
        all_y_vals.append(np.array(y_vals))

    all_y_vals = np.vstack(all_y_vals)
    mean, stddev = np.mean(all_y_vals, axis=0), np.std(all_y_vals, axis=0)

    all_data = np.vstack((x, mean, mean - stddev, mean + stddev))
    cdf = pd.DataFrame(all_data.T, columns=['Objective',
                                            'Threshold Percentage (Mean)',
                                            'Threshold Percentage (Min)',
                                            'Threshold Percentage (Max)'])

    return cdf

def index_of(env_name):
    return list(shared_params.keys()).index(env_name)


def plot_cdf_data(algo_dataframes: list, alg_data_dirs: dict, archive_type: str, reevaluated_archives=False, axs=None, **kwargs):
    '''
    :param algorithm: name of the algorithm
    :param alg_data_dirs: contains env: path string-string pairs for all envs for this algorithm
    :param archive_type: either pyribs or qdax depending on which repo produced the archive
    :param reevaluated_archives: whether to plot corrected QD metrics or not
    :param axs: matplotlib axes objects
    '''
    algorithm = "policy_diffusion"
    standalone_plot = False
    if axs is None:
        standalone_plot = True
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    subtitle = 'Archive CDFs'
    prefix = 'Corrected ' if reevaluated_archives else ''
    title = prefix + subtitle

    for i, (exp_name, env_dir) in enumerate(alg_data_dirs.items()):
        base_cfg = AttrDict(shared_params[exp_name])
        base_cfg['title'] = exp_name

        cfg = copy.copy(base_cfg)
        cfg.update({'archive_dir': env_dir, 'algorithm': algorithm})
        # dataframe_fn = get_ppga_df if archive_type == 'pyribs' else get_pgame_df
        # algo_dataframes = dataframe_fn(env_dir, reevaluated_archives)
        algo_cdf = compile_cdf(cfg, dataframes=algo_dataframes)

        if standalone_plot:
            (j, k) = np.unravel_index(i, (2, 2))  # b/c there are 4 envs
            make_cdf_plot(cfg, algo_cdf, axs[j][k])
        else:
            env_idx = index_of(exp_name)
            make_cdf_plot(cfg, algo_cdf, axs[3][env_idx])


def reevaluate_ppga_archive_mem(env_cfg: AttrDict,
                            normalize_obs: bool,
                            normalize_returns: bool,
                            original_archive: GridArchive,
                            solution_batch_size: int = 200,
                            reconstructed_agents: bool = False,
                            vae: nn.Module = None,
                            sampler=None,
                            scale_factor=None,
                            diffusion_model=None,
                            save_path=None,
                            center_data: bool = False,
                            weight_normalizer = None,
                            uniform_sampling = False,
                            latent_shape = None,
                            average = False,
                            ):
    num_sols = len(original_archive)
    print(f'{num_sols=}')
    envs_per_agent = 50
    env_cfg.env_batch_size = envs_per_agent * solution_batch_size
    vec_env = make_vec_env_brax(env_cfg)

    obs_shape, action_shape = vec_env.single_observation_space.shape, vec_env.single_action_space.shape
    device = torch.device('cuda')

    if vae is not None:
        vae.to(device)

    if diffusion_model is not None:
        diffusion_model.to(device)

    if reconstructed_agents:
        assert vae is not None and isinstance(vae, nn.Module), 'reconstructed_agents was set to true, but a valid VAE ' \
                                                               'model was not passed in'

    agents = []
    measures_list = []
    for elite in original_archive:
        agent = Actor(obs_shape[0], action_shape, normalize_obs, normalize_returns).deserialize(elite.solution).to(
            device)
        if normalize_obs:
            obs_norm = elite.metadata['obs_normalizer']
            if isinstance(obs_norm, dict):
                agent.obs_normalizer.load_state_dict(obs_norm)
            else:
                agent.obs_normalizer = obs_norm
        agents.append(agent)
        measures_list.append(elite.measures)
    agents = np.array(agents)
    measures_list = np.array(measures_list)

    all_objs, all_measures, all_metadata = [], [], []
    true_measures = []
    for i in range(0, num_sols, solution_batch_size):
        agent_batch = agents[i: i + solution_batch_size]
        measure_batch = measures_list[i: i + solution_batch_size]

        if reconstructed_agents:
            agent_batch = reconstruct_agents_from_ldm(agent_batch, measure_batch, vae, device, sampler,
                            scale_factor, diffusion_model,
                            center_data=center_data,
                            weight_normalizer=weight_normalizer,
                            latent_shape=latent_shape,
                            uniform_sampling=uniform_sampling
                        )

        if env_cfg.env_batch_size % len(agent_batch) != 0 and len(original_archive) % solution_batch_size != 0:
            del vec_env
            env_cfg.env_batch_size = len(agent_batch) * envs_per_agent
            vec_env = make_vec_env_brax(env_cfg)
        # print(f'Evaluating solution batch {i}')
        vec_inference = VectorizedActor(agent_batch, Actor, obs_shape=obs_shape, action_shape=action_shape,
                                        normalize_obs=normalize_obs, normalize_returns=normalize_returns,
                                        deterministic=True).to(device)
        objs, measures, metadata = evaluate(vec_inference, vec_env, env_cfg.num_dims, normalize_obs=normalize_obs, average=average)
        all_objs.append(objs)
        all_measures.append(measures)
        all_metadata.append(metadata)
        true_measures.extend(measure_batch)

    all_objs, all_measures = np.concatenate(all_objs).reshape(1, -1).mean(axis=0), \
        np.concatenate(all_measures).reshape(1, -1, env_cfg.num_dims).mean(axis=0)
    all_metadata = np.concatenate(all_metadata).reshape(-1)
    true_measures = np.stack(true_measures)

    # print(f'{all_objs.shape=}, {all_measures.shape=}')
    # Measure_Error_Mean is the 2 norm of the difference between the true measure and the estimated measure
    if average:
        Measure_Error_Mean = np.linalg.norm(true_measures - all_measures, axis=1).mean()
    else:
        Measure_Error_Mean = 0

    # create a new archive
    archive_dims = [env_cfg.grid_size] * env_cfg.num_dims
    bounds = [(0., 1.0) for _ in range(env_cfg.num_dims)]
    new_archive = GridArchive(solution_dim=1,
                              dims=archive_dims,
                              ranges=bounds,
                              threshold_min=-10000,
                              seed=env_cfg.seed,
                              qd_offset=reward_offset[env_cfg.env_name])
    all_objs[np.isnan(all_objs)] = 0
    # add the re-evaluated solutions to the new archive
    new_archive.add(
        np.ones((len(all_objs), 1)),
        all_objs,
        all_measures,
        all_metadata
    )
    # print(f'Coverage: {new_archive.stats.coverage} \n'
    #       f'Max fitness: {new_archive.stats.obj_max} \n'
    #       f'Avg Fitness: {new_archive.stats.obj_mean} \n'
    #       f'QD Score: {new_archive.offset_qd_score}')

    # if save_path is not None:
    #     archive_fp = os.path.join(save_path, f'{env_cfg.env_name}_reeval_archive.pkl')
    #     with open(archive_fp, 'wb') as f:
    #         pickle.dump(new_archive, f)
    # archive_df = new_archive.as_pandas(include_solutions=True)

    return {
        "Measure_Error_Mean":Measure_Error_Mean, 
        "Coverage":new_archive.stats.coverage, 
        "Max_Fitness":new_archive.stats.obj_max,
        "Avg_Fitness":new_archive.stats.obj_mean,
        "QD_Score":new_archive.offset_qd_score,
        "Archive":new_archive.as_pandas(include_solutions=False)
    }


def evaluate_ldm_subsample_with_mem(env_name: str, archive_df=None, ldm=None, autoencoder=None, N: int = 100,
                           image_path: str = None, suffix: str = None, ignore_first: bool = False, sampler=None,
                           scale_factor=None, clip_obs_rew: bool = False,
                            normalize_obs: bool = False,
                            uniform_sampling: bool = False,
                            center_data: bool = False,
                            latent_shape = None,
                            weight_normalizer = None,
                            cut_out: bool = False,
                            average: bool = False,):
    if type(archive_df) == str:
        with open(archive_df, 'rb') as f:
            archive_df = pickle.load(f)
    else:
        archive_df = archive_df


    env_cfg = AttrDict(shared_params[env_name]['env_cfg'])
    env_cfg.seed = 1111
    env_cfg.clip_obs_rew = clip_obs_rew

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

    normalize_obs, normalize_returns = True, False
    archive_info = None
    if not ignore_first:
        print('Re-evaluated Original Archive')
        archive_info = reevaluate_ppga_archive_mem(env_cfg,
                                                               normalize_obs,
                                                               normalize_returns,
                                                               original_archive,
                                                               average=average)
        # Archive_Measure_Error_Mean = archive_info['Measure_Error_Mean']

    print('Re-evaluated Reconstructed Archive')
    recon_info = reevaluate_ppga_archive_mem(env_cfg,
                                            normalize_obs,
                                            normalize_returns,
                                            original_archive,
                                            reconstructed_agents=True,
                                            vae=autoencoder,
                                            sampler=sampler,
                                            scale_factor=scale_factor,
                                            diffusion_model=ldm,
                                            center_data=center_data,
                                            uniform_sampling=uniform_sampling,
                                            weight_normalizer=weight_normalizer,
                                            latent_shape = latent_shape,
                                            average=average,
                                            )
    # Measure_Error_Mean = recon_info['Measure_Error_Mean']
    # reconstructed_results = {
    #     'Coverage': reconstructed_evaluated_archive.stats.coverage,
    #     'Max_fitness': reconstructed_evaluated_archive.stats.obj_max,
    #     'Avg_Fitness': reconstructed_evaluated_archive.stats.obj_mean,
    #     'QD_Score': reconstructed_evaluated_archive.offset_qd_score
    # }
    # results = {
    #     'Original': original_results if not ignore_first else None,
    #     'Reconstructed': reconstructed_results,
    # }

    return recon_info, archive_info



if __name__ == '__main__':
    final_results_folder = "/home/shashank/research/qd"
    paper_results_folder = "paper_results2"
    envs = ["halfcheetah", "ant", "walker2d", "humanoid",] #["halfcheetah", "ant", "walker2d", "humanoid"]
    continue_from_previous = False
    averaging = True


    latent_channels = 4
    emb_channels=4
    z_height=4
    z_channels=4
    enc_fc_hid=64
    obsnorm_hid=64
    # archive_data_path="data/humanoid/archive100x100.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batch_size = 32
    Num = 5000



    for env in envs:
        existing_experiments_dict_path = f"{final_results_folder}/{paper_results_folder}/{env}/diffusion_model/experiments_dict.json"
        if os.path.exists(existing_experiments_dict_path):
            with open(existing_experiments_dict_path, 'rb') as f:
                experiments_dict = json.load(f)
                
        experiments_dict_cpy = copy.deepcopy(experiments_dict)
        (env_ind, env_experiments_dict) = [(ind,env_dict) for ind,env_dict in enumerate(experiments_dict['results']) if env_dict['env'] == env][0]

        obs_dim, action_shape = shared_params[env]['obs_dim'], np.array([shared_params[env]['action_dim']])

        timesteps = 600
        logvar = torch.full(fill_value=0., size=(timesteps,))

        for exp_ind, experiment in enumerate(env_experiments_dict['experiments_dict_list']):
            centering = experiment["centering"]
            ghn_hid = experiment["ghn_size"]
            dataframes = []
            orig_dataframes = []
            for folder_ind, folder in enumerate(experiment["folders"]):

                seed = folder["seed"]
                name = folder["name"]

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                args_json_file = f"{final_results_folder}/{paper_results_folder}/{env}/diffusion_model/{name}/args.json"
                with open(args_json_file, 'r') as f:
                    args = json.load(f)

                model = ConditionalUNet(
                    in_channels=latent_channels,
                    out_channels=latent_channels,
                    channels=64,
                    n_res_blocks=1,
                    attention_levels=[],
                    channel_multipliers=[1, 2, 4],
                    n_heads=4,
                    d_cond=256,
                    logvar=logvar,
                    measure_dim=shared_params[env]['env_cfg']['num_dims']
                )
                # model_path = f"{final_results_folder}/{args['model_checkpoint_folder']}/{name}.pt"
                model_path = f"{final_results_folder}/{paper_results_folder}/{env}/diffusion_model/{name}/model_checkpoints/{name}.pt"
                model.load_state_dict(torch.load(model_path))
                model.to(device)

                autoencoder = AutoEncoder(emb_channels=emb_channels,
                    z_channels=z_channels,
                    obs_shape=obs_dim,
                    action_shape=action_shape,
                    z_height=z_height,
                    ghn_hid=ghn_hid,
                    enc_fc_hid = enc_fc_hid,
                    obsnorm_hid=obsnorm_hid,
                )     
                autoencoder_path = f"{final_results_folder}/{args['autoencoder_cp_path']}"
                autoencoder.load_state_dict(torch.load(autoencoder_path))
                autoencoder.to(device)
                
                betas = cosine_beta_schedule(timesteps)
                gauss_diff = LatentDiffusion(betas, num_timesteps=timesteps, device=device)
                sampler = DDIMSampler(gauss_diff, n_steps=100)

                # print size of models
                print(f"\nGHN size: {ghn_hid}")
                print(f"Unet size: {sum(p.numel() for p in model.parameters())}")
                print(f"Autoencoder decoder size: {sum(p.numel() for p in autoencoder.decoder.parameters())}")
                print(f"total size: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in autoencoder.decoder.parameters())}")

                if "Measure_Error_Mean" in folder.keys() and continue_from_previous:
                    print(f"Skipping {folder}")
                    continue

                weight_normalizer = None
                train_dataloader, train_archive, weight_normalizer = shaped_elites_dataset_factory(
                    env, batch_size=train_batch_size, is_eval=False,
                    center_data=centering,
                    use_language=False,
                    N=Num,
                    weight_normalizer=weight_normalizer)
        
                gt_params_batch, measures = next(iter(train_dataloader))
                with torch.no_grad():
                    batch = autoencoder.encode(gt_params_batch).sample().detach()
                    # rescale the embeddings to be unit variance
                    std = batch.flatten().std()
                    scale_factor = 1. / std
                    scale_factor = scale_factor.item()
                    
                dataset_kwargs = {
                    'center_data': centering,
                    'weight_normalizer': weight_normalizer
                }

                recon_info, archive_info = evaluate_ldm_subsample_with_mem(env_name=env,
                    archive_df=train_archive[0],
                    ldm=model,
                    autoencoder=autoencoder,
                    N=-1,
                    image_path = None,
                    # suffix = str(epoch),
                    ignore_first=True,
                    sampler=sampler,
                    scale_factor=scale_factor,
                    normalize_obs=True,
                    clip_obs_rew=True,
                    uniform_sampling = False,
                    cut_out=False,
                    average=averaging,
                    latent_shape = (z_channels, z_height, z_height),
                    **dataset_kwargs
                )

                Measure_Error_Mean = recon_info['Measure_Error_Mean']
                Recon_Coverage = recon_info['Coverage']
                Recon_Avg_Fitness = recon_info['Avg_Fitness']
                Recon_QD_Score = recon_info['QD_Score']
                print(f"Measure_Error_Mean: {Measure_Error_Mean}")

                experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Measure_Error_Mean'] = Measure_Error_Mean
                experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Recon_Coverage'] = Recon_Coverage
                experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Recon_Avg_Fitness'] = Recon_Avg_Fitness
                experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Recon_QD_Score'] = Recon_QD_Score
                

                if archive_info is not None:
                    Archive_Coverage = archive_info['Coverage']
                    Archive_Measure_Error_Mean = archive_info['Measure_Error_Mean']
                    Archive_Avg_Fitness = archive_info['Avg_Fitness']
                    Archive_QD_Score = archive_info['QD_Score']

                    experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Archive_Measure_Error_Mean'] = Archive_Measure_Error_Mean
                    experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Archive_Coverage'] = Archive_Coverage
                    experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Archive_Avg_Fitness'] = Archive_Avg_Fitness
                    experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Archive_QD_Score'] = Archive_QD_Score

                with open(f"{final_results_folder}/{paper_results_folder}/{env}/diffusion_model/experiments_dict.json", 'w') as f:
                    json.dump(experiments_dict_cpy, f, indent=4)

                dataframes.append(recon_info["Archive"])
                orig_dataframes.append(train_archive[0])
            
            
            base_cfg = AttrDict(shared_params[env])
            base_cfg['title'] = env

            cfg = copy.copy(base_cfg)
            cfg.update({'archive_dir': None, 'algorithm': "policy_diffusion"})
            # dataframe_fn = get_ppga_df if archive_type == 'pyribs' else get_pgame_df
            # algo_dataframes = dataframe_fn(env_dir, reevaluated_archives)
            algo_cdf = compile_cdf(cfg, dataframes=dataframes)
            orig_cdf = compile_cdf(cfg, dataframes=orig_dataframes)

            # save cdf as csv
            algo_cdf.to_csv(f"{final_results_folder}/{paper_results_folder}/{env}/diffusion_model/cdf_{centering}.csv")
            orig_cdf.to_csv(f"{final_results_folder}/{paper_results_folder}/{env}/diffusion_model/cdf_orig_{centering}.csv")

            env_idx = index_of(env)
            fig, axs = plt.subplots(1, 1, figsize=(6, 6))
            make_cdf_plot(cfg, algo_cdf, axs, original=False, standalone=True)
            make_cdf_plot(cfg, orig_cdf, axs, original=True, standalone=True)



            # save the figure
            fig.savefig(f"{final_results_folder}/{paper_results_folder}/{env}/diffusion_model/cdf_{centering}.png")