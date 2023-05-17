import os
from pathlib import Path
import torch
import pickle
import json
import numpy as np
import matplotlib
matplotlib.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 20,
    }
)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from typing import Optional
from diffusion.gaussian_diffusion import cosine_beta_schedule, linear_beta_schedule, GaussianDiffusion
from diffusion.latent_diffusion import LatentDiffusion
from diffusion.ddim import DDIMSampler
from autoencoders.policy.hypernet import HypernetAutoEncoder as AutoEncoder
from dataset.shaped_elites_dataset import WeightNormalizer
from attrdict import AttrDict
from utils.tensor_dict import TensorDict, cat_tensordicts
from RL.actor_critic import Actor
from utils.normalize import ObsNormalizer
from models.cond_unet import ConditionalUNet, LangConditionalUNet
from envs.brax_custom.brax_env import make_vec_env_brax
from utils.brax_utils import shared_params, rollout_many_agents
from algorithm.train_autoencoder import shaped_elites_dataset_factory
from autoencoders.policy.hypernet import HypernetAutoEncoder, ModelEncoder
from tqdm import tqdm


# params to config
device = torch.device('cuda')
env_name = 'humanoid'
seed = 1111
normalize_obs = True
normalize_rewards = False
obs_shape = shared_params[env_name]['obs_dim']
action_shape = np.array([shared_params[env_name]['action_dim']])
mlp_shape = (128, 128, action_shape)
train_batch_size=32
latent_diffusion = True
use_ddim = True
center_data = True
use_language = True
latent_channels = 4
latent_size = 4
timesteps = 600


env_cfg = AttrDict({
    'env_name': env_name,
    'env_batch_size': None,
    'num_dims': 2,
    'seed': seed,
    'num_envs': 1,
    'clip_obs_rew': True,
})



# paths to VAE and diffusion model checkpoint
autoencoder_path = '/home/shashank/research/qd/paper_language_results/humanoid/autoencoder/humanoid_autoencoder_20230503-072924_111/model_checkpoints/humanoid_autoencoder_20230503-072924_111.pt'
model_path = '/home/shashank/research/qd/paper_language_results/humanoid/diffusion_model/humanoid_diffusion_model_20230515-032333_0/model_checkpoints/humanoid_diffusion_model_20230515-032333_0.pt'
weight_normalizer_path = 'results/humanoid/weight_normalizer.pkl'

weight_normalizer = None
dataloader, train_archive, weight_normalizer = shaped_elites_dataset_factory(env_name,
                                                                                batch_size=train_batch_size,
                                                                                is_eval=False,
                                                                                center_data=center_data,
                                                                                cut_out=False,
                                                                                weight_normalizer=weight_normalizer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HypernetAutoEncoder(emb_channels=4,
                            z_channels=4,
                            obs_shape=obs_shape,
                            action_shape=action_shape,
                            z_height=4,
                            conditional=False,
                            ghn_hid=32,
                            enc_fc_hid = 64,
                            obsnorm_hid=64,
                            )
model.load_state_dict(torch.load(autoencoder_path, map_location=device))
model.to(device)
model.eval()

# get the latent representation of the dataset by getting the mean of the posterior
latent_dataset = []
measures_0 = []
measures_1 = []
for step, (policies, measures) in tqdm(enumerate(dataloader)):
    _, posterior = model(policies)
    latent_dataset.append(posterior.mean.flatten(1).detach().cpu().numpy())
    measures_0.append(measures[:,0].detach().cpu().numpy())
    measures_1.append(measures[:,1].detach().cpu().numpy())

latent_dataset = np.concatenate(latent_dataset, axis=0)
measures_0 = np.concatenate(measures_0, axis=0)
measures_1 = np.concatenate(measures_1, axis=0)

# use tsne to visualize the latent space
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=5, n_jobs=-1)
latent_dataset_tsne = tsne.fit_transform(latent_dataset)

# plot the latent space for both measures
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].scatter(latent_dataset_tsne[:,0], latent_dataset_tsne[:,1], c=measures_0, cmap='viridis')
ax[0].set_title('Measure 0')
# ax[0].set_xlabel('Latent dim 0')
# ax[0].set_ylabel('Latent dim 1')
# add colorbar to the plot
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=measures_0.min(), vmax=measures_0.max()))
sm._A = []
cbar = fig.colorbar(sm, ax=ax[0])


ax[1].scatter(latent_dataset_tsne[:,0], latent_dataset_tsne[:,1], c=measures_1, cmap='viridis')
ax[1].set_title('Measure 1')
# ax[1].set_xlabel('Latent dim 0')
# ax[1].set_ylabel('Latent dim 1')
# add colorbar to the plot
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=measures_1.min(), vmax=measures_1.max()))
sm._A = []
cbar = fig.colorbar(sm, ax=ax[1])


plt.savefig('latent_space.png')
plt.close()