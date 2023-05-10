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
from utils.archive_utils import reconstruct_agents_from_vae, evaluate
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
                    "centering": False,
                    "ghn_size": 16,
                    "kl": 1e-2,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "conditional_humanoid_autoencoder_20230508-224726_111"
                        },
                        {
                            "seed": 222,
                            "name": "conditional_humanoid_autoencoder_20230508-224831_222"
                        },
                        {
                            "seed": 333,
                            "name": "conditional_humanoid_autoencoder_20230509-020842_333"
                        },
                        {
                            "seed": 444,
                            "name": "conditional_humanoid_autoencoder_20230509-021417_444"
                        },
                    ],

                },

                {
                    "centering": False,
                    "ghn_size": 16,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "conditional_humanoid_autoencoder_20230504-154350_111"
                        },
                        {
                            "seed": 222,
                            "name": "conditional_humanoid_autoencoder_20230504-154350_222"
                        },
                        {
                            "seed": 333,
                            "name": "conditional_humanoid_autoencoder_20230504-154351_333"
                        },
                        {
                            "seed": 444,
                            "name": "conditional_humanoid_autoencoder_20230504-154351_444"
                        },
                    ],

                },

                {
                    "centering": True,
                    "ghn_size": 16,
                    "kl": 1e-2,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "conditional_humanoid_autoencoder_20230508-223935_111"
                        },
                        {
                            "seed": 222,
                            "name": "conditional_humanoid_autoencoder_20230508-224058_222"
                        },
                        {
                            "seed": 333,
                            "name": "conditional_humanoid_autoencoder_20230508-224214_333"
                        },
                        {
                            "seed": 444,
                            "name": "conditional_humanoid_autoencoder_20230508-224703_444"
                        },
                    ],

                },

                {
                    "centering": True,
                    "ghn_size": 16,
                    "kl": 1e-6,
                    "folders": [
                        {
                            "seed": 111,
                            "name": "conditional_humanoid_autoencoder_20230504-154109_111"
                        },
                        {
                            "seed": 222,
                            "name": "conditional_humanoid_autoencoder_20230504-162654_222"
                        },
                        {
                            "seed": 333,
                            "name": "conditional_humanoid_autoencoder_20230504-154341_333"
                        },
                        {
                            "seed": 444,
                            "name": "conditional_humanoid_autoencoder_20230504-154350_444"
                        },
                    ],
                },
            ]
        },
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

    # if diffusion_model is not None:
    #     diffusion_model.to(device)

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
            agent_batch = reconstruct_agents_from_vae(agent_batch, vae, device,
                                                          center_data=center_data,
                                                          weight_normalizer=weight_normalizer,
                                                          measure_batch = measure_batch,)

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

    return Measure_Error_Mean


def evaluate_vae_subsample_with_mem(env_name: str, archive_df=None, autoencoder=None, N: int = 100,
                           image_path: str = None, suffix: str = None, ignore_first: bool = False, clip_obs_rew: bool = False,
                            normalize_obs: bool = False,
                            uniform_sampling: bool = False,
                            center_data: bool = False,
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
    Archive_Measure_Error_Mean = 0
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
                                                              center_data=center_data,
                                                              uniform_sampling=uniform_sampling,
                                                              weight_normalizer=weight_normalizer,
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




final_results_folder = "/home/shashank/research/qd"
env = "humanoid"
latent_channels = 4
emb_channels=4
z_height=4
z_channels=4
enc_fc_hid=64
obsnorm_hid=64
# archive_data_path="data/humanoid/archive100x100.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_batch_size = 32

existing_experiments_dict_path = f"{final_results_folder}/paper_results/{env}/autoencoder/experiments_dict.json"
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
    for folder_ind, folder in enumerate(experiment["folders"]):
        if "Measure_Error_Mean" in folder.keys():
            print(f"Skipping {folder}")
            continue

        seed = folder["seed"]
        name = folder["name"]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        args_json_file = f"{final_results_folder}/paper_results/{env}/autoencoder/{name}/args.json"
        with open(args_json_file, 'r') as f:
            args = json.load(f)

        autoencoder = AutoEncoder(emb_channels=emb_channels,
            z_channels=z_channels,
            obs_shape=obs_dim,
            action_shape=action_shape,
            z_height=z_height,
            ghn_hid=ghn_hid,
            enc_fc_hid = enc_fc_hid,
            obsnorm_hid=obsnorm_hid,
            conditional = True
        )     
        autoencoder_path = f"{final_results_folder}/{args['model_checkpoint_folder']}/{name}.pt"
        autoencoder.load_state_dict(torch.load(autoencoder_path))
        autoencoder.to(device)
        
    
        weight_normalizer = None
        _, train_archive, weight_normalizer = shaped_elites_dataset_factory(
            env, batch_size=train_batch_size, is_eval=False,
            center_data=centering,
            use_language=False,
            weight_normalizer=weight_normalizer)
  

        dataset_kwargs = {
            'center_data': centering,
            'weight_normalizer': weight_normalizer
        }

        Measure_Error_Mean, Archive_Measure_Error_Mean = evaluate_vae_subsample_with_mem(env_name=env,
            archive_df=train_archive[0],
            autoencoder=autoencoder,
            N=5000,
            image_path = None,
            # suffix = str(epoch),
            ignore_first=True,
            normalize_obs=True,
            clip_obs_rew=True,
            uniform_sampling = False,
            cut_out=False,
            average=True,
            **dataset_kwargs
        )

        print(f"Measure_Error_Mean: {Measure_Error_Mean}")
        experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Measure_Error_Mean'] = Measure_Error_Mean
        experiments_dict_cpy['results'][env_ind]['experiments_dict_list'][exp_ind]['folders'][folder_ind]['Archive_Measure_Error_Mean'] = Archive_Measure_Error_Mean
        del train_archive
        with open(f"{final_results_folder}/paper_results/{env}/autoencoder/experiments_dict.json", 'w') as f:
            json.dump(experiments_dict_cpy, f, indent=4)