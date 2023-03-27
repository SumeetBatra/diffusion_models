import torch

from RL.actor_critic import Actor
from envs.brax_custom.brax_env import make_vec_env_brax
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

    avg_rew = total_reward.mean()
    avg_measures = measures.mean(dim=0)

    rew_std = total_reward.std()
    measures_std = measures.std(dim=0)

    return avg_rew, avg_measures, rew_std, measures_std


def compare_rec_to_gt_policy(gt_agent, rec_agent, env_cfg, vec_env, device, deterministic=True):
    # get a ground truth policy and evaluate it. Then get the reconstructed policy and compare its
    # performance and behavior to the ground truth

    gt_reward, gt_measures, gt_rew_std, gt_measures_std = rollout_agent(gt_agent, env_cfg, vec_env, device, deterministic)
    log.debug(f'Ground Truth Agent -- Reward: {gt_reward.item()} +/- {gt_rew_std.item()} , '
              f'\n Measures: {gt_measures.detach().cpu().numpy()} +/- {gt_measures_std.detach().cpu().numpy()}')

    rec_reward, rec_measures, rec_rew_std, rec_measures_std = rollout_agent(rec_agent, env_cfg, vec_env, device, deterministic)
    log.debug(f'Reconstructed Agent -- Reward: {rec_reward.item()} +/- {rec_rew_std.item()}, '
              f'\n Measures: {rec_measures.detach().cpu().numpy()} +/- {rec_measures_std.detach().cpu().numpy()}')

    perf_error = torch.abs(gt_reward - rec_reward) / gt_reward
    measure_error = torch.norm(gt_measures - rec_measures) / torch.norm(gt_measures)

    log.debug(f" Normalized Performance Difference: {perf_error.item()}, \n Normalized Measure Difference: {measure_error.item()}")
