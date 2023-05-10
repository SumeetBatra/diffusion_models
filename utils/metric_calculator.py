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

experiments_dict = {
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
        }
    ]
}



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
        print(f'Evaluating solution batch {i}')
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
    Measure_Error_Mean = np.linalg.norm(true_measures - all_measures, axis=1).mean()

    # # create a new archive
    # archive_dims = [env_cfg.grid_size] * env_cfg.num_dims
    # bounds = [(0., 1.0) for _ in range(env_cfg.num_dims)]
    # new_archive = GridArchive(solution_dim=1,
    #                           dims=archive_dims,
    #                           ranges=bounds,
    #                           threshold_min=-10000,
    #                           seed=env_cfg.seed,
    #                           qd_offset=reward_offset[env_cfg.env_name])
    # all_objs[np.isnan(all_objs)] = 0
    # # add the re-evaluated solutions to the new archive
    # new_archive.add(
    #     np.ones((len(all_objs), 1)),
    #     all_objs,
    #     all_measures,
    #     all_metadata
    # )
    # print(f'Coverage: {new_archive.stats.coverage} \n'
    #       f'Max fitness: {new_archive.stats.obj_max} \n'
    #       f'Avg Fitness: {new_archive.stats.obj_mean} \n'
    #       f'QD Score: {new_archive.offset_qd_score}')

    # if save_path is not None:
    #     archive_fp = os.path.join(save_path, f'{env_cfg.env_name}_reeval_archive.pkl')
    #     with open(archive_fp, 'wb') as f:
    #         pickle.dump(new_archive, f)

    return Measure_Error_Mean


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
    if not ignore_first:
        print('Re-evaluated Original Archive')
        Archive_Measure_Error_Mean = reevaluate_ppga_archive_mem(env_cfg,
                                                               normalize_obs,
                                                               normalize_returns,
                                                               original_archive,
                                                               average=average)

    print('Re-evaluated Reconstructed Archive')
    Measure_Error_Mean = reevaluate_ppga_archive_mem(env_cfg,
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

    return Measure_Error_Mean, Archive_Measure_Error_Mean




final_results_folder = "./"
experiments_dict_cpy = copy.deepcopy(experiments_dict)
env = "walker2d"
(env_ind, env_experiments_dict) = [(ind,env_dict) for ind,env_dict in enumerate(experiments_dict['results']) if env_dict['env'] == env][0]
latent_channels = 4
emb_channels=4
z_height=4
z_channels=4
enc_fc_hid=64
obsnorm_hid=64
archive_data_path="data/humanoid/archive100x100.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_batch_size = 32

existing_experiments_dict_path = f"{final_results_folder}/paper_results/{env}/diffusion_model/experiments_dict.pkl"
if os.path.exists(existing_experiments_dict_path):
    with open(existing_experiments_dict_path, 'rb') as f:
        experiments_dict = pickle.load(f)
        
obs_dim, action_shape = shared_params[env]['obs_dim'], np.array([shared_params[env]['action_dim']])

timesteps = 600
logvar = torch.full(fill_value=0., size=(timesteps,))

for exp_ind, experiment in enumerate(env_experiments_dict['experiments_dict_list']):
    centering = experiment["centering"]
    ghn_hid = experiment["ghn_size"]
    for folder_ind, folder in enumerate(experiment["folders"]):
        if "Measure_Error_Mean" in folder.keys():
            print(f"Skipping {folder}")
            continue

        seed = folder["seed"]
        name = folder["name"]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        args_json_file = f"{final_results_folder}/paper_results/{env}/diffusion_model/{name}/args.json"
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
            logvar=logvar
        )
        model_path = f"{final_results_folder}/{args['model_checkpoint_folder']}/{name}.pt"
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
    
        weight_normalizer = None
        train_dataloader, train_archive, weight_normalizer = shaped_elites_dataset_factory(
            env, batch_size=train_batch_size, is_eval=False,
            center_data=centering,
            use_language=False,
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

        Measure_Error_Mean, Archive_Measure_Error_Mean = evaluate_ldm_subsample_with_mem(env_name=env,
            archive_df=train_archive[0],
            ldm=model,
            autoencoder=autoencoder,
            N=-1,
            image_path = None,
            # suffix = str(epoch),
            ignore_first=False,
            sampler=sampler,
            scale_factor=scale_factor,
            normalize_obs=True,
            clip_obs_rew=True,
            uniform_sampling = False,
            cut_out=False,
            average=True,
            latent_shape = (z_channels, z_height, z_height),
            **dataset_kwargs
        )

        print(f"Measure_Error_Mean: {Measure_Error_Mean}")
        experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Measure_Error_Mean'] = Measure_Error_Mean
        experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Archive_Measure_Error_Mean'] = Archive_Measure_Error_Mean

        with open(f"{final_results_folder}/paper_results/{env}/diffusion_model/experiments_dict.json", 'w') as f:
            json.dump(experiments_dict_cpy, f, indent=4)