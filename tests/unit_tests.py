import torch
import numpy as np

from RL.actor_critic import Actor
from dataset.policy_dataset import preprocess_model, postprocess_model


def test_pre_post_process():
    '''Make sure that the model tensor -> policy method returns
    the exact same policy as the original policy '''

    # define inputs
    obs_shape, action_shape = 18, np.array(6)

    agent = Actor(obs_shape=obs_shape, action_shape=action_shape, normalize_obs=False, normalize_returns=False)
    mlp_shape = (128, 128, 6)

    model_tensor = preprocess_model(agent, mlp_shape)

    # define a new agent that we will use to reconstruct the original one
    agent_rec = Actor(obs_shape, action_shape, normalize_obs=False, normalize_returns=False)
    agent_rec = postprocess_model(agent_rec, agent, model_tensor, mlp_shape)

    orig_params, rec_params = agent.serialize(), agent_rec.serialize()
    orig_params = torch.from_numpy(orig_params)
    rec_params = torch.from_numpy(rec_params)

    assert torch.allclose(orig_params, rec_params)