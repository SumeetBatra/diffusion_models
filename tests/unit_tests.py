import torch
import numpy as np

from RL.actor_critic import Actor
from dataset.shaped_elites_dataset import preprocess_model, postprocess_model


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