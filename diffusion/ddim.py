'''impl based off of https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html'''
import torch
import numpy as np

from tqdm import tqdm
from functools import partial
from typing import List, Optional
from diffusion.gaussian_diffusion import GaussianDiffusion


class DDIMSampler():
    ''' https://arxiv.org/pdf/2010.02502.pdf'''
    def __init__(self, diffusion_model: GaussianDiffusion, n_steps: int, ddim_discretize: str = 'uniform', ddim_eta: float = 0):
        self.n_steps = diffusion_model.num_timesteps
        self.diffusion_model = diffusion_model
        if ddim_discretize == 'uniform':
            # subsample every T/S steps, where T is the num_timesteps used during training, and S is the num
            # steps to take when sampling, where S < T
            c = self.n_steps // n_steps
            self.timesteps = np.asarray(list(range(0, self.n_steps, c))) + 1
        elif ddim_discretize == 'quad':
            self.timesteps = ((np.linspace(0, np.sqrt(n_steps * 0.8), n_steps))**2).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)

        with torch.no_grad():
            self.betas = diffusion_model.betas
            self.alphas_cumprod = diffusion_model.alphas_cumprod
            self.alphas_cumprod_prev = diffusion_model.alphas_cumprod_prev

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.sqrt_alphas_cumprod = diffusion_model.sqrt_alphas_cumprod
            self.sqrt_one_minus_alphas_cumprod = diffusion_model.sqrt_one_minus_alphas_cumprod
            self.log_one_minus_alphas_cumprod = torch.log(self.sqrt_one_minus_alphas_cumprod)
            self.sqrt_recip_alphas_cumprod = diffusion_model.sqrt_recip_alphas_cumprod
            self.sqrt_recipm1_alphas_cumprod = diffusion_model.sqrt_recipm1_alphas_cumprod

            # ddim sampling params
            self.ddim_alpha = self.alphas_cumprod[self.timesteps]
            self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
            self.ddim_alpha_prev = torch.cat([self.alphas_cumprod[0:1], self.alphas_cumprod[self.timesteps[:-1]]])
            self.ddim_sigma = (ddim_eta *
                               ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                                (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** 0.5
                               )
            self.ddim_sqrt_one_minus_alpha = (1. - self.ddim_alpha) ** 0.5

    @torch.no_grad()
    def sample(self,
               model: torch.nn.Module,
               shape: List[int],
               cond: torch.Tensor,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: Optional[torch.Tensor] = None,
               uncond_scale: float = 1.,
               uncond_cond: Optional[torch.Tensor] = None,
               skip_steps: int = 0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bs = shape[0]
        x = x_last if x_last is not None else torch.randn(shape, device=device)
        timesteps = np.flip(self.timesteps)[skip_steps:]

        for i, step in enumerate(timesteps):
            index = len(timesteps) - i - 1
            ts = x.new_full(size=(bs,), fill_value=step, dtype=torch.long)
            x, pred_x0, e_t = self.p_sample(model, x, cond, ts, step, index=index, repeat_noise=repeat_noise,
                                            temperature=temperature, uncond_scale=uncond_scale, uncond_cond=uncond_cond)
        return x

    @torch.no_grad()
    def p_sample(self,
                 model: torch.nn.Module,
                 x: torch.Tensor,
                 c: torch.Tensor,
                 t: torch.Tensor,
                 step: int,
                 index: int,
                 *,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1.,
                 uncond_cond: Optional[torch.Tensor] = None):
        e_t = model(x, t)
        x_prev, pred_x0 = self.get_x_prev_and_pred_x0(e_t, index, x, temperature=temperature, repeat_noise=repeat_noise)
        return x_prev, pred_x0, e_t

    def get_x_prev_and_pred_x0(self,
                               e_t: torch.Tensor,
                               index: int,
                               x: torch.Tensor,
                               temperature: float,
                               repeat_noise: bool):
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = self.ddim_sqrt_one_minus_alpha[index]
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / (alpha ** 0.5)

        dir_xt = (1. - alpha_prev - sigma ** 2).sqrt() * e_t

        if sigma == 0.:
            noise = 0.
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        else:
            noise = torch.randn(x.shape, device=x.device)

        noise *= temperature
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev, pred_x0

