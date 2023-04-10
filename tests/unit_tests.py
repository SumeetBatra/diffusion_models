import copy

import torch
import numpy as np

from RL.actor_critic import Actor
from dataset.tensor_elites_dataset import preprocess_model, postprocess_model
from utils.normalize import ObsNormalizer


def test_pre_post_process():
    '''Make sure that the model tensor -> policy method returns
    the exact same policy as the original policy '''

    # define inputs
    obs_shape, action_shape = 18, np.array(6)

    agent = Actor(obs_shape=obs_shape, action_shape=action_shape, normalize_obs=False, normalize_returns=False)
    agent.actor_mean[0].bias.data = torch.range(0, 127)
    agent.actor_mean[2].bias.data = torch.randn_like(agent.actor_mean[2].bias.data)
    agent.actor_mean[4].bias.data = torch.range(0, 5)
    mlp_shape = (128, 128, 6)

    model_tensor = preprocess_model(agent, mlp_shape)

    # define a new agent that we will use to reconstruct the original one
    agent_rec = Actor(obs_shape, action_shape, normalize_obs=False, normalize_returns=False)
    agent_rec = postprocess_model(agent_rec, model_tensor, mlp_shape, deterministic=False)

    orig_params, rec_params = agent.serialize(), agent_rec.serialize()
    orig_params = torch.from_numpy(orig_params)
    rec_params = torch.from_numpy(rec_params)

    assert torch.allclose(orig_params, rec_params)


def test_merge_obsnorm():
    obs_shape, action_shape = 18, np.array(6)

    obs = torch.randn(100, obs_shape)

    normalizer = ObsNormalizer(obs_shape)
    mean, var = torch.randn(obs_shape), torch.abs(torch.randn(obs_shape))
    normalizer.obs_rms.mean = mean
    normalizer.obs_rms.var = var

    agent = Actor(obs_shape, action_shape, normalize_obs=True, normalize_returns=False)
    agent.obs_normalizer = normalizer

    def integrate_obs_normalizer(weights_dict, obs_normalizer, inp_coef):
        w_in = weights_dict['actor_mean.0.weight']
        b_in = weights_dict['actor_mean.0.bias']
        mean, var = obs_normalizer.obs_rms.mean, obs_normalizer.obs_rms.var

        w_new = inp_coef * (w_in / torch.sqrt(var + 1e-8))
        b_new = inp_coef * (b_in - (mean / torch.sqrt(var + 1e-8)) @ w_in.T)
        weights_dict['actor_mean.0.weight'] = w_new
        weights_dict['actor_mean.0.bias'] = b_new
        return weights_dict

    merged_agent = copy.deepcopy(agent)
    res_state_dict = integrate_obs_normalizer(merged_agent.state_dict(), normalizer, 1.0)
    merged_agent.load_state_dict(res_state_dict)

    norm_obs = (obs - mean) / (torch.sqrt(var) + 1e-9)
    out1 = agent(norm_obs).flatten()
    out2 = merged_agent(obs).flatten()

    assert torch.allclose(out1, out2)