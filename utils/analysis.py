import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt

from attrdict import AttrDict
from autoencoders.policy.hypernet import HypernetAutoEncoder as VAE
from collections import OrderedDict
from ribs.archives import GridArchive, CVTArchive
from utils.brax_utils import rollout_many_agents
from utils.archive_utils import archive_df_to_archive, reevaluate_ppga_archive, save_heatmap
from utils.brax_utils import shared_params, rollout_many_agents, js_divergence
from envs.brax_custom import reward_offset
from models.cond_unet import ConditionalUNet
from diffusion.latent_diffusion import LatentDiffusion
from diffusion.gaussian_diffusion import cosine_beta_schedule
from diffusion.ddim import DDIMSampler
from RL.actor_critic import Actor
from RL.vectorized import VectorizedActor
from envs.brax_custom.brax_env import make_vec_env_brax


def evaluate_vae_subsample(env_name: str, archive_df=None, model=None, N: int = 100, image_path: str = None,
                            suffix: str = None, ignore_first: bool = False, clip_obs_rew: bool = False,
                            normalize_obs: bool = False,
                            center_data: bool = False,
                            weight_normalizer = None,):

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
        original_reevaluated_archive = reevaluate_ppga_archive(env_cfg,
                                                               normalize_obs,
                                                               normalize_returns,
                                                               original_archive)
        print('Re-evaluated Original Archive')
        original_results = {
            'Coverage': original_reevaluated_archive.stats.coverage,
            'Max_fitness': original_reevaluated_archive.stats.obj_max,
            'Avg_Fitness': original_reevaluated_archive.stats.obj_mean,
            'QD_Score': original_reevaluated_archive.offset_qd_score
        }

    reconstructed_evaluated_archive = reevaluate_ppga_archive(env_cfg,
                                                              normalize_obs,
                                                              normalize_returns,
                                                              original_archive,
                                                              reconstructed_agents=True,
                                                              vae=vae,
                                                              center_data=center_data,
                                                              weight_normalizer=weight_normalizer,)
    print('Re-evaluated Reconstructed Archive')
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
            orig_image_array = save_heatmap(original_reevaluated_archive,
                                            os.path.join(image_path, f"original_archive_{suffix}.png"))
        recon_image_array = save_heatmap(reconstructed_evaluated_archive,
                                         os.path.join(image_path, f"reconstructed_archive_{suffix}.png"))

    image_results = {
        'Original': orig_image_array if not ignore_first else None,
        'Reconstructed': recon_image_array,
    }
    return results, image_results


def evaluate_ldm_subsample(env_name: str, archive_df=None, ldm=None, autoencoder=None, N: int = 100,
                           image_path: str = None, suffix: str = None, ignore_first: bool = False, sampler=None,
                           scale_factor=None):
    if type(archive_df) == str:
        with open(archive_df, 'rb') as f:
            archive_df = pickle.load(f)
    else:
        archive_df = archive_df

    env_cfg = AttrDict(shared_params[env_name]['env_cfg'])
    env_cfg.seed = 1111

    if N != -1:
        archive_df = archive_df.sample(N)
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
                                                              vae=autoencoder,
                                                              sampler=sampler,
                                                              scale_factor=scale_factor,
                                                              diffusion_model=ldm,
                                                              )
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
            orig_image_array = save_heatmap(original_reevaluated_archive,
                                            os.path.join(image_path, f"original_archive_{suffix}.png"))
        recon_image_array = save_heatmap(reconstructed_evaluated_archive,
                                         os.path.join(image_path, f"reconstructed_archive_{suffix}.png"))

    image_results = {
        'Original': orig_image_array if not ignore_first else None,
        'Reconstructed': recon_image_array,
    }
    return results, image_results


def initialize_all_models_and_archives(env_name):
    obs_shape, action_shape = shared_params[env_name]['obs_dim'], np.array(shared_params[env_name]['action_dim'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init the archive
    archive_df_path = f'data/{env_name}/archive_100x100.pkl'
    with open(archive_df_path, 'rb') as f:
        archive_df = pickle.load(f)

    vae_cp = f'./checkpoints/autoencoder_{env_name}.pt'
    vae = VAE(emb_channels=4, z_channels=4, obs_shape=obs_shape, action_shape=action_shape, z_height=4)
    vae.load_state_dict(torch.load(vae_cp))
    vae.to(device)
    vae.eval()

    unet_model_cp = './checkpoints/diffusion_model_walker2d_20230415-1447.pt'
    timesteps = 600
    logvar = torch.full(fill_value=0., size=(timesteps,))
    model = ConditionalUNet(in_channels=4,
                            out_channels=4,
                            channels=64,
                            n_res_blocks=1,
                            attention_levels=[],
                            channel_multipliers=[1, 2, 4],
                            n_heads=4,
                            d_cond=256,
                            logvar=logvar)
    model.load_state_dict(torch.load(unet_model_cp))
    model.to(device)
    model.eval()

    betas = cosine_beta_schedule(timesteps)
    diffusion = LatentDiffusion(betas, num_timesteps=timesteps, device=device)
    sampler = DDIMSampler(diffusion, n_steps=100)

    cfg_path = './checkpoints/cfg.json'
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
        cfg = AttrDict(cfg)

    return archive_df, vae, model, sampler, cfg


def get_agents_with_measures(sampler: DDIMSampler,
                             model: ConditionalUNet,
                             vae: VAE,
                             measures: torch.Tensor,
                             scale_factor: float,
                             classifier_scale: float,
                             classifier_free_guidance: bool = True,
                             latent: bool = False):
    '''Sample policy with desired measure from the diffusion model'''
    samples = sampler.sample(model, shape=[measures.shape[0], 4, 4, 4], cond=measures,
                            classifier_free_guidance=classifier_free_guidance, classifier_scale=classifier_scale)
    if latent:
        return samples
    samples = samples * (1 / scale_factor)
    samples = vae.decode(samples)
    return samples


def classifier_free_guidance_hyperparameter_search():
    '''
    Evaluate the performance of the model over different values for classifier scale on points sampled from the archive
    Sampling scheme: Divide the archive into 4 quadrants and uniformly sample k solutions from each quadrant.
    :return: Average measure error in reconstructed policies
    '''
    classifier_scales = [0, 1, 2, 4, 8, 16]
    env_name = 'walker2d'
    obs_shape, action_shape = shared_params[env_name]['obs_dim'], np.array(shared_params[env_name]['action_dim'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # how many times to run this experiment. More trials = lower uncertainty
    num_trials = 10

    # number of random measures to sample
    sample_size = 50
    env_cfg = AttrDict({
        'env_name': env_name,
        'env_batch_size': 100 * sample_size,
        'num_dims': 2,
        'seed': 0,
    })
    vec_env = make_vec_env_brax(env_cfg)

    archive_df, vae, model, sampler, cfg = initialize_all_models_and_archives(env_name)

    scale_factor = cfg.scale_factor

    measures = torch.rand(2 * sample_size).reshape(sample_size, 2)
    measures = measures.to(device)

    rewards, losses, js_divs = np.zeros(len(classifier_scales)), np.zeros(len(classifier_scales)), np.zeros(len(classifier_scales))
    for t in range(num_trials):
        print(f'Trial {t}')
        for i, cs in enumerate(classifier_scales):
            use_guidance = True
            if cs == 0:
                use_guidance = False
            agents = get_agents_with_measures(sampler, model, vae, measures, scale_factor, cs, classifier_free_guidance=use_guidance)
            for agent in agents:
                agent.actor_logstd = nn.Parameter(torch.zeros(action_shape)).to(device)

            rews, res_measures = rollout_many_agents(agents, env_cfg, vec_env, device, verbose=False)
            avg_reward = rews.mean()

            # calculate the js divergence b/w original and resulting measures
            gt_mean, gt_cov = measures.mean(0).detach().cpu().numpy(), measures.T.cov().detach().cpu().numpy()
            res_mean, res_cov = res_measures.mean(0).detach().cpu().numpy(), res_measures.T.cov().detach().cpu().numpy()
            js_div = js_divergence(gt_mean, gt_cov, res_mean, res_cov)

            # calculate the mse b/w original and res measures
            measure_mse = nn.functional.mse_loss(measures, res_measures).item()

            print(f'Classifier scale: {cs}, {avg_reward=}, {js_div=}, {measure_mse=}')
            rewards[i] += avg_reward
            losses[i] += measure_mse
            js_divs[i] += js_div

    rewards /= num_trials
    losses /= num_trials
    js_divs /= num_trials

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(classifier_scales, rewards)
    ax[1].plot(classifier_scales, losses)
    ax[2].plot(classifier_scales, js_divs)
    ax[0].set_title(f'Average Reward Over {num_trials} Trials')
    ax[1].set_title(f'Average Measure MSE Over {num_trials} Trials')
    ax[2].set_title(f'Average JS Divergence Over {num_trials} Trials')
    fig.tight_layout()
    plt.show()


def evaluate_measure_distance_cvt():
    num_cells = 50
    env_name = 'walker2d'
    archive_df, vae, model, sampler, cfg = initialize_all_models_and_archives(env_name)
    scale_factor = cfg.scale_factor
    env_cfg = AttrDict(shared_params[env_name]['env_cfg'])
    env_cfg.seed = 0
    env_cfg.env_batch_size = num_cells * 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vec_env = make_vec_env_brax(env_cfg)

    def compute_centroids():
        solution_dim = archive_df.filter(regex='solution*').to_numpy().shape[1]
        ranges = [(0.0, 1.0)] * env_cfg.num_dims

        cvt_archive = CVTArchive(solution_dim=solution_dim,
                                 cells=num_cells,
                                 ranges=ranges,
                                 seed=env_cfg.seed)
        return cvt_archive.centroids

    centroids = compute_centroids()
    centroids = torch.from_numpy(centroids).to(torch.float32).to(device)

    classifier_scale = 1.0
    agents = get_agents_with_measures(sampler, model, vae, centroids, scale_factor, classifier_scale)
    for agent in agents:
        agent.actor_logstd = nn.Parameter(torch.zeros(shared_params[env_name]['action_dim'])).to(device)

    rews, res_measures = rollout_many_agents(agents, env_cfg, vec_env, device)
    measure_mse = nn.functional.mse_loss(centroids, res_measures).item()

    gt_mean, gt_cov = centroids.mean(0).detach().cpu().numpy(), centroids.T.cov().detach().cpu().numpy()
    res_mean, res_cov = res_measures.mean(0).detach().cpu().numpy(), res_measures.T.cov().detach().cpu().numpy()
    js_div = js_divergence(gt_mean, gt_cov, res_mean, res_cov)

    print(f'{measure_mse=}, {js_div=}')


if __name__ == '__main__':
    archive_df_path = '/home/sumeet/QDPPO/experiments/ppga_halfcheetah_adaptive_stddev_no_obs_norm/1111/checkpoints/' \
                      'cp_00001990/archive_df_00001990.pkl'

    model_path = 'checkpoints/autoencoder.pt'

    env_name = 'halfcheetah'

    # evaluate_vae_subsample(env_name, archive_df_path, model_path)
    evaluate_measure_distance_cvt()