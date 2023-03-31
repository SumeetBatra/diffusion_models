import torch
import scipy.stats as stats
import numpy as np

from RL.actor_critic import Actor
from utils.utilities import log
from RL.vectorized import VectorizedActor


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


def compare_rec_to_gt_policy(gt_agent, rec_agent, env_cfg, vec_env, device, deterministic=True):
    # get a ground truth policy and evaluate it. Then get the reconstructed policy and compare its
    # performance and behavior to the ground truth

    gt_rewards, gt_measures = rollout_agent(gt_agent, env_cfg, vec_env, device, deterministic)
    # log.debug(f'Ground Truth Agent -- Reward: {gt_rewards.mean().item()} +/- {gt_rewards.std().item()} , '
            #   f'\n Measures: {gt_measures.mean(dim=0).detach().cpu().numpy()} +/- {gt_measures.mean(dim=0).std().detach().cpu().numpy()}')

    rec_rewards, rec_measures = rollout_agent(rec_agent, env_cfg, vec_env, device, deterministic)
    # log.debug(f'Reconstructed Agent -- Reward: {rec_rewards.mean().item()} +/- {rec_rewards.std().item()}, '
            #   f'\n Measures: {rec_measures.mean(dim=0).detach().cpu().numpy()} +/- {rec_measures.mean(dim=0).std().detach().cpu().numpy()}')

    ttest_res = stats.ttest_ind(gt_measures.detach().cpu().numpy(), rec_measures.detach().cpu().numpy(), equal_var=False)
    return {'t_test': ttest_res,
            'measure_mse': torch.square(gt_measures - rec_measures).mean(0).detach().cpu().numpy(),
            'Rewards/original': gt_rewards.mean().item(),
            'Measures/original': gt_measures.mean(dim=0).detach().cpu().numpy(),
            'Rewards/reconstructed': rec_rewards.mean().item(),
            'Measures/reconstructed': rec_measures.mean(dim=0).detach().cpu().numpy(),
            }


def rollout_many_agents(agents: list[Actor], env_cfg, vec_env, device, deterministic=True, verbose=True):
    '''
    Evaluate multiple agents multiple times
    :returns: Sum rewards and average measures for all agents
    '''
    # TODO: with obs_norm enabled is not tested in needs testing!

    assert vec_env.num_envs % len(agents) == 0, 'The num_envs parameter must be a multiple of the number of agents'

    normalize_obs = True if agents[0].obs_normalizer is not None else False
    obs_shape = vec_env.single_observation_space.shape

    vec_agent = VectorizedActor(agents, Actor, normalize_obs=normalize_obs, obs_shape=obs_shape, normalize_returns=False)

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
        # stack all the obs means and vars from all policies
        obs_means = np.array([agent.obs_normalizer.obs_rms.mean for agent in agents])
        obs_vars = np.array([agent.obs_normalizer.obs_rms.var for agent in agents])
        obs_means = torch.from_numpy(obs_means).to(device)
        obs_vars = torch.from_numpy(obs_vars).to(device)

    while not torch.all(dones):
        with torch.no_grad():
            if normalize_obs:
                obs = (obs - obs_means) / (torch.sqrt(obs_vars) + 1e-8)
            if deterministic:
                acts = vec_agent.actor_mean(obs)
            else:
                acts, _, _ = vec_agent.get_action(obs)

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
        dim=1).detach().cpu().numpy()

    total_reward = total_reward.reshape((vec_agent.num_models, vec_env.num_envs // vec_agent.num_models)).mean(
        axis=1)

    max_reward = np.max(total_reward)
    min_reward = np.min(total_reward)
    mean_reward = np.mean(total_reward)
    mean_traj_length = torch.mean(traj_lengths.to(torch.float64)).detach().cpu().numpy().item()
    objective_measures = np.concatenate((total_reward.reshape(-1, 1), measures), axis=1)

    if verbose:
        np.set_printoptions(suppress=True)
        log.debug('Finished Evaluation Step')
        log.info(f'Reward + Measures: {objective_measures}')
        log.info(f'Max Reward on eval: {max_reward}')
        log.info(f'Min Reward on eval: {min_reward}')
        log.info(f'Mean Reward across all agents: {mean_reward}')
        log.info(f'Average Trajectory Length: {mean_traj_length}')

    return total_reward.reshape(-1, ), measures.reshape(-1, env_cfg.num_dims)