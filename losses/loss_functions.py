import torch
import torch.nn.functional as F

from diffusion.gaussian_diffusion import GaussianDiffusion


def p_losses(denoise_model, x_start, t, gaussian_diffusion: GaussianDiffusion, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = gaussian_diffusion.q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss