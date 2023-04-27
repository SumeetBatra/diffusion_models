import torch
import scipy.stats as stats
import numpy as np

from collections import OrderedDict
from RL.actor_critic import Actor
from utils.utilities import log
from RL.vectorized import VectorizedActor


shared_params = OrderedDict({
    'walker2d':
        {
            'objective_range': (0, 5000),
            'objective_resolution': 100,
            'archive_resolution': 2500,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'obs_dim': 17,
            'action_dim': 6,
            'env_cfg': {
                'env_name': 'walker2d',
                'num_dims': 2,
                'episode_length': 1000,
                'grid_size': 100
            }
        },
    'halfcheetah':
        {
            'objective_range': (0, 9000),
            'objective_resolution': 100,
            'archive_resolution': 2500,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'obs_dim': 18,
            'action_dim': 6,
            'env_cfg': {
                'env_name': 'halfcheetah',
                'num_dims': 2,
                'episode_length': 1000,
                'grid_size': 100
            }
        },
    'humanoid':
        {
            'objective_range': (0, 10000),
            'objective_resolution': 100,
            'archive_resolution': 2500,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'obs_dim': 227,
            'action_dim': 17,
            'env_cfg': {
                'env_name': 'humanoid',
                'num_dims': 2,
                'episode_length': 1000,
                'grid_size': 100
            }
        },
    'ant':
        {
            'objective_range': (0, 7000),
            'objective_resolution': 100,
            'archive_resolution': 10000,
            'skip_len': 200,
            'algorithm_name': 'cma_mae_100_0.01',
            'obs_dim': 87,
            'action_dim': 8,
            'env_cfg': {
                'env_name': 'ant',
                'num_dims': 4,
                'episode_length': 1000,
                'grid_size': 10,
            }
        }
})


def kl_divergence(mu1, cov1, mu2, cov2):
    """
    Calculates the KL divergence between two Gaussian distributions.

    Parameters:
    mu1 (numpy array): Mean of the first Gaussian distribution
    cov1 (numpy array): Covariance matrix of the first Gaussian distribution
    mu2 (numpy array): Mean of the second Gaussian distribution
    cov2 (numpy array): Covariance matrix of the second Gaussian distribution

    Returns:
    KL divergence (float)
    """

    # calculate KL divergence using formula
    kl_div = 0.5 * (np.trace(np.linalg.inv(cov2).dot(cov1)) +
                    np.dot((mu2 - mu1).T, np.dot(np.linalg.inv(cov2), (mu2 - mu1))) -
                    len(mu1) + np.log(np.linalg.det(cov2) / (np.linalg.det(cov1) + 1e-9)))

    return kl_div


def js_divergence(mu1, cov1, mu2, cov2):
    '''Jensen-Shannon symmetric divergence metric. It is assumed that all variables
    mu1/2 and cov1/2 parametrize two normal distributions, otherwise this calculation is incorrect'''

    # sum of gaussians is gaussian
    mu_m, cov_m = 0.5 * (mu1 + mu2), 0.5 * (cov1 + cov2)

    res = 0.5 * (kl_divergence(mu1, cov1, mu_m, cov_m) + kl_divergence(mu2, cov2, mu_m, cov_m))
    return res


def rollout_agent(agent: Actor, env_cfg, vec_env, device, deterministic=True):
    if agent.obs_normalizer is not None:
        obs_mean, obs_var = agent.obs_normalizer.obs_rms.mean, agent.obs_normalizer.obs_rms.var

    num_steps = 1000
    traj_length = 0
    total_reward = torch.zeros(vec_env.num_envs)

    obs = vec_env.reset()
    obs = obs.to(device)

    dones = torch.BoolTensor([False for _ in range(vec_env.num_envs)])
    all_dones = torch.zeros((num_steps, vec_env.num_envs)).to(device)
    measures_acc = torch.zeros((num_steps, vec_env.num_envs, env_cfg.num_dims)).to(device)
    measures = torch.zeros((vec_env.num_envs, env_cfg.num_dims)).to(device)

    while not torch.all(dones):
        with torch.no_grad():
            if agent.obs_normalizer is not None:
                obs = (obs - obs_mean) / (torch.sqrt(obs_var) + 1e-8)
            if deterministic:
                acts = agent.actor_mean(obs)
            else:
                acts, _, _ = agent.get_action(obs)

            obs, rew, next_dones, infos = vec_env.step(acts)
            measures_acc[traj_length] = infos['measures']
            obs = obs.to(device)
            total_reward += rew.detach().cpu().numpy() * ~dones.cpu().numpy()
            dones = torch.logical_or(dones, next_dones.cpu())
            all_dones[traj_length] = dones.long().clone()
            traj_length += 1

    # the first done in each env is where that trajectory ends
    traj_lengths = torch.argmax(all_dones, dim=0) + 1

    for i in range(vec_env.num_envs):
        measures[i] = measures_acc[:traj_lengths[i], i].sum(dim=0) / traj_lengths[i]

    return total_reward, measures


def calculate_statistics(gt_rews, gt_measures, rec_rewards, rec_measures): 
    '''
    Calculate various statistics based on batches of rewards and measures evaluated from the ground truth 
    and reconstructed policies
    '''
    gt_mean, gt_cov = gt_measures.mean(0), np.cov(gt_measures.T)
    rec_mean, rec_cov = rec_measures.mean(0), np.cov(rec_measures.T)
    js_div = js_divergence(gt_mean, gt_cov, rec_mean, rec_cov)

    ttest_res = stats.ttest_ind(gt_measures, rec_measures, equal_var=False)

    return {
        'js_div': js_div,
        't_test': ttest_res,
        'measure_mse': np.square(gt_measures.mean(0) - rec_measures.mean(0)),

        'Rewards/original': gt_rews.mean().item(),
        'Measures/original_mean': gt_measures.mean(axis=0),
        'Measures/original_std': gt_measures.std(axis=0),

        'Rewards/reconstructed': rec_rewards.mean().item(),
        'Measures/reconstructed_mean': rec_measures.mean(axis=0),
        'Measures/reconstructed_std': rec_measures.std(axis=0),
    }


def compare_rec_to_gt_policy(gt_agent, rec_agent, env_cfg, vec_env, device, deterministic=True):
    # get a ground truth policy and evaluate it. Then get the reconstructed policy and compare its
    # performance and behavior to the ground truth

    gt_rewards, gt_measures = rollout_agent(gt_agent, env_cfg, vec_env, device, deterministic)
    # log.debug(f'Ground Truth Agent -- Reward: {gt_rewards.mean().item()} +/- {gt_rewards.std().item()} , '
            #   f'\n Measures: {gt_measures.mean(dim=0).detach().cpu().numpy()} +/- {gt_measures.mean(dim=0).std().detach().cpu().numpy()}')

    rec_rewards, rec_measures = rollout_agent(rec_agent, env_cfg, vec_env, device, deterministic)
    # log.debug(f'Reconstructed Agent -- Reward: {rec_rewards.mean().item()} +/- {rec_rewards.std().item()}, '
            #   f'\n Measures: {rec_measures.mean(dim=0).detach().cpu().numpy()} +/- {rec_measures.mean(dim=0).std().detach().cpu().numpy()}')
    
    stats = calculate_statistics(gt_rewards.detach().cpu().numpu(), 
                                 gt_measures.detach().cpu().numpy(),
                                 rec_rewards.detach().cpu().numpy(),
                                 rec_measures.detach().cpu().numpy())

    return stats 


def rollout_many_agents(agents: list[Actor], env_cfg, vec_env, device, deterministic=True, verbose=False, normalize_obs = False):
    '''
    Evaluate multiple agents multiple times
    :returns: Sum rewards and average measures for all agents
    '''
    # TODO: with obs_norm enabled is not tested in needs testing!

    assert vec_env.num_envs % len(agents) == 0, 'The num_envs parameter must be a multiple of the number of agents'

    num_envs_per_agent = vec_env.num_envs // len(agents)

    # normalize_obs = False
    obs_shape = vec_env.single_observation_space.shape

    vec_agent = VectorizedActor(agents, Actor, normalize_obs=False, obs_shape=obs_shape, normalize_returns=False)

    total_reward = np.zeros(vec_env.num_envs)
    traj_length = 0
    num_steps = 1000

    obs = vec_env.reset()
    obs = obs.to(device)
    dones = torch.BoolTensor([False for _ in range(vec_env.num_envs)])
    all_dones = torch.zeros((num_steps, vec_env.num_envs)).to(device)
    measures_acc = torch.zeros((num_steps, vec_env.num_envs, env_cfg.num_dims)).to(device)
    measures = torch.zeros((vec_env.num_envs, env_cfg.num_dims)).to(device)

    if normalize_obs:
        obs_means = torch.cat([agent.obs_normalizer.obs_rms.mean.repeat(num_envs_per_agent,1) for agent in agents]).to(device)
        obs_vars = torch.cat([agent.obs_normalizer.obs_rms.var.repeat(num_envs_per_agent,1) for agent in agents]).to(device)

    while not torch.all(dones):
        with torch.no_grad():
            if normalize_obs:
                obs = (obs - obs_means) / (torch.sqrt(obs_vars) + 1e-8)
            if deterministic:
                acts = vec_agent.actor_mean(obs)
            else:
                acts, _, _ = vec_agent.get_action(obs)
            acts = acts.to(torch.float32)
            obs, rew, next_dones, infos = vec_env.step(acts)
            measures_acc[traj_length] = infos['measures']
            obs = obs.to(device)
            total_reward += rew.detach().cpu().numpy() * ~dones.cpu().numpy()
            dones = torch.logical_or(dones, next_dones.cpu())
            all_dones[traj_length] = dones.long().clone()
            traj_length += 1

    # the first done in each env is where that trajectory ends
    traj_lengths = torch.argmax(all_dones, dim=0) + 1
    for i in range(vec_env.num_envs):
        measures[i] = measures_acc[:traj_lengths[i], i].sum(dim=0) / traj_lengths[i]
    measures = measures.reshape(vec_agent.num_models, vec_env.num_envs // vec_agent.num_models, -1).mean(
        dim=1)

    total_reward = total_reward.reshape((vec_agent.num_models, vec_env.num_envs // vec_agent.num_models)).mean(
        axis=1)

    max_reward = np.max(total_reward)
    min_reward = np.min(total_reward)
    mean_reward = np.mean(total_reward)
    mean_traj_length = torch.mean(traj_lengths.to(torch.float64)).detach().cpu().numpy().item()
    objective_measures = np.concatenate((total_reward.reshape(-1, 1), measures.detach().cpu().numpy()), axis=1)

    if verbose:
        np.set_printoptions(suppress=True)
        log.debug('Finished Evaluation Step')
        log.info(f'Reward + Measures: {objective_measures}')
        log.info(f'Max Reward on eval: {max_reward}')
        log.info(f'Min Reward on eval: {min_reward}')
        log.info(f'Mean Reward across all agents: {mean_reward}')
        log.info(f'Average Trajectory Length: {mean_traj_length}')

    return total_reward.reshape(-1, ), measures.reshape(-1, env_cfg.num_dims).detach().cpu().numpy()