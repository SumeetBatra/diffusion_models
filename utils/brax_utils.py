import torch
import scipy.stats as stats

from RL.actor_critic import Actor
from utils.utilities import log


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
    log.debug(f'Ground Truth Agent -- Reward: {gt_rewards.mean().item()} +/- {gt_rewards.std().item()} , '
              f'\n Measures: {gt_measures.mean(dim=0).detach().cpu().numpy()} +/- {gt_measures.mean(dim=0).std().detach().cpu().numpy()}')

    rec_rewards, rec_measures = rollout_agent(rec_agent, env_cfg, vec_env, device, deterministic)
    log.debug(f'Reconstructed Agent -- Reward: {rec_rewards.mean().item()} +/- {rec_rewards.std().item()}, '
              f'\n Measures: {rec_measures.mean(dim=0).detach().cpu().numpy()} +/- {rec_measures.mean(dim=0).std().detach().cpu().numpy()}')

    ttest_res = stats.ttest_ind(gt_measures.detach().cpu().numpy(), rec_measures.detach().cpu().numpy(), equal_var=False)
    return {'t_test': ttest_res,
            'Rewards/original': gt_rewards.mean().item(),
            'Measures/original': gt_measures.mean(dim=0).detach().cpu().numpy(),
            'Rewards/reconstructed': rec_rewards.mean().item(),
            'Measures/reconstructed': rec_measures.mean(dim=0).detach().cpu().numpy()}
