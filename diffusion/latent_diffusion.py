import torch

from diffusion.gaussian_diffusion import GaussianDiffusion
from losses.loss_functions import mse


class LatentDiffusion(GaussianDiffusion):
    def __init__(self, betas, num_timesteps, device):
        super().__init__(betas, num_timesteps, device)

        self.clip_denoised = False

        self.vlb_weights = self.betas ** 2 / (2 * self.posterior_variance * self.alphas * (1 - self.alphas_cumprod))
        self.vlb_weights[0] = self.vlb_weights[1]

        logvar = torch.full(fill_value=0., size=(num_timesteps,))
        self.logvar = torch.nn.Parameter(logvar, requires_grad=True).to(self.device)

    def compute_training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)

        model_output = model(x_t, t)  # TODO: implement conditioning via model_kwargs
        target = noise

        #  See https://arxiv.org/pdf/2112.10752.pdf Section B on why we can simplify vlb loss like this
        vlb_loss = mse(model_output, target)
        vlb_weights = self.vlb_weights.to(self.device)
        vlb_loss = (vlb_weights[t] * vlb_loss).mean()
        vlb_loss *= self.num_timesteps / 1000.

        # simple loss term
        logvar_t = self.logvar[t]
        simple_loss = mse(model_output, target)
        simple_loss = (simple_loss / torch.exp(logvar_t)) + logvar_t

        loss = simple_loss + vlb_loss
        return loss

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        model_output = model(x, t)  # TODO: Implement model conditioning via model_kwargs

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            'mean': model_mean,
            'variance': posterior_variance,
            'log_variance': posterior_log_variance,
            'pred_xstart': pred_xstart
        }