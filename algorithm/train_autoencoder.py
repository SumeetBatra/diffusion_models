import torch
import numpy as np
import os
import pickle
import glob
import torch.nn.functional as F

from pathlib import Path
from torch.optim import Adam
from dataset.policy_dataset import ElitesDataset, postprocess_model
from torch.utils.data import DataLoader
from autoencoders.policy.resnet3d import ResNet3DAutoEncoder
from autoencoders.policy.transformer import TransformerPolicyAutoencoder
from autoencoders.policy.hypernet import HypernetAutoEncoder
from RL.actor_critic import Actor


def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)


def dataset_factory():
    archive_data_path = '/home/sumeet/diffusion_models/data'
    archive_dfs = []

    archive_df_paths = glob.glob(archive_data_path + '/archive*100x100*.pkl')
    for path in archive_df_paths:
        with open(path, 'rb') as f:
            archive_df = pickle.load(f)
            archive_dfs.append(archive_df)

    mlp_shape = (128, 128, 6)

    dummy_agent = Actor(obs_shape=18, action_shape=np.array([6]))

    elite_dataset = ElitesDataset(archive_dfs, mlp_shape, dummy_agent)

    return DataLoader(elite_dataset, batch_size=32, shuffle=True)


def mse_loss_from_unpadded_params(policy_in_tensors, rec_agents):
    bs = policy_in_tensors.shape[0]

    # convert reconstructed actors to params tensor
    rec_params = np.array([agent.serialize() for agent in rec_agents])
    rec_params = torch.from_numpy(rec_params).reshape(bs, -1)

    mlp_shape = (128, 128, 6)
    dummy_agent = Actor(obs_shape=18, action_shape=np.array([6]))
    # first convert the data from padded -> unpadded params tensors
    gt_actor_params = torch.Tensor([postprocess_model(dummy_agent, tensor, mlp_shape, return_model=False, deterministic=True) for tensor in policy_in_tensors]).reshape(bs, -1)

    return F.mse_loss(gt_actor_params, rec_params)


def train_autoencoder():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_checkpoint = None
    model = HypernetAutoEncoder(emb_channels=8, z_channels=4)
    if model_checkpoint is not None:
        print(f'Loading model from checkpoint')
        model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    mse_loss_func = mse_loss_from_unpadded_params
    kl_loss_coef = 1e-4

    model_checkpoint_folder = Path('./checkpoints')
    model_checkpoint_folder.mkdir(exist_ok=True)

    dataloader = dataset_factory()

    disc_start = 50001
    kl_weight = 1e-6
    disc_weight = 0.5
    # loss_func = LPIPSWithDiscriminator(disc_start, kl_weight=kl_weight, disc_weight=disc_weight)
    # optimizer2 = Adam(loss_func.discriminator.parameters(), lr=1e-3)

    epochs = 20
    global_step = 0
    for epoch in range(epochs):
        print(f'{epoch=}')
        print(f'{global_step=}')

        for step, (policies, measures) in enumerate(dataloader):
            optimizer.zero_grad()

            policies = policies.to(device)
            measures = measures.to(device)

            rec_policies, posterior = model(policies)
            # loss = loss_func(batch, img_out, posterior, global_step, 0)
            # loss += loss_func(batch, img_out, posterior, global_step, 1)
            loss = mse_loss_func(policies, rec_policies) + kl_loss_coef * posterior.kl().mean()

            loss.backward()
            if step % 100 == 0:
                print(f'Loss: {loss.item()}')
                # print(f'grad norm: {grad_norm(model)}') TODO: fix this
            optimizer.step()
            global_step += step

    print('Saving final model checkpoint...')
    torch.save(model.state_dict(), os.path.join(str(model_checkpoint_folder), 'autoencoder.pt'))


if __name__ == '__main__':
    train_autoencoder()

