#  https://huggingface.co/blog/annotated-diffusion
#  https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from losses.loss_functions import normal_kl, discretized_gaussian_log_likelihood


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class GaussianDiffusion:
    '''
    Class for forward and reverse diffusion process
    '''

    def __init__(self, betas, num_timesteps, device):
        self.betas = betas
        self.num_timesteps = num_timesteps
        self.device = device

        # variables that will be reused later i.e. to calculate noise at intermediate timesteps etc.

        # define alphas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # we can compute this directly for any intermediate timestep b/c sum of gaussians is gaussian,
        # giving us a closed form solution
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        # variance is 0 at beginning of diffusion chain so we "clip" it by replacing the 0th index with the 1st
        self.posterior_log_variance_clipped = torch.log(torch.cat((self.posterior_variance[1].view(-1, 1), self.posterior_variance[1:].view(-1, 1))).squeeze())

        # equation 11 first term in improved DDPM
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # equation 11 second term in Improved DDPMs
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps

    def _compute_vlb_loss(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        '''
        Compute the variational lower bound loss
        :return: dict w/ following keys:
        'output': shape [N] tensor of NLLs or KLs
        'pred_xstart': the x_0 predictions
        '''
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = kl.mean(dim=list(range(1, len(kl.shape)))) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = decoder_nll.mean(dim=list(range(1, len(decoder_nll.shape)))) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], 'log_variance': out['log_variance']}

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        '''
        Compute the mean and variance of the diffusion posterior:
                q(x_{t-1} | x_t, x_0)
        '''
        assert x_start.shape == x_t.shape
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + \
                         extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        assert (
                posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        '''
        Apply the model to get p(x_{t-1} | x_t) as well as prediction of initial x_0
        :param model: model takes in batch of timesteps and signal
        :param x: the [N x C x ...] tensor at time t
        :param t: 1-D tensor of timesteps
        :param clip_denoised: if True, clip the denoised signal into [-1, 1]
        :param denoised_fn: If not None, a function which applies to the x_start prediction before it is used to sample.
        Applies before clip_denoised
        :param model_kwargs: If not None, a dict of extra keyword args to pass to the model. Can be used for conditioning
        :return: a dict with keys:
        'mean': model mean ouput
        'variance': model variance output
        'log_variance': the log of 'variance'
        'pred_xstart': the prediction for x_0
        '''

        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t)  # TODO: Implement model conditioning via model_kwargs

        # assume that we learn the variance within a range given by B_t and Bhat_t (posterior variance)
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = torch.split(model_output, C, dim=1)

        # learn the variance between a range
        min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = extract(torch.log(self.betas), t, x.shape)
        # the model_var_values is [-1, 1] for [min_var, max_var]. Need to shift it to [0, 1] range
        frac = (model_var_values + 1) / 2
        # equation 15 in Improved DDPM
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # assume that we predict epsilon noise
        pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_xstart': pred_xstart
        }

    def compute_training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        # assume rescaled MSE loss for noise mean output and that we learn the variance
        model_output = model(x_t, t)  # TODO: implement conditioning via model_kwargs

        B, C = x_t.shape[:2]
        assert model_output.shape == (B, C * 2, *x_t.shape[2:])  # TODO: make sure this is true
        model_output, model_var_values = torch.split(model_output, C, dim=1)
        # Learn the variance using the variational bound, but don't let
        # it affect our mean prediction.
        frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
        vlb_loss_dict = self._compute_vlb_loss(model=lambda *args, r=frozen_out: r,
                                          x_start=x_start,
                                          x_t=x_t,
                                          t=t,
                                          clip_denoised=False)
        vlb_loss = vlb_loss_dict['output']
        model_log_var = vlb_loss_dict['log_variance']
        # Divide by 1000 for equivalence with initial implementation (Equation 16 term 2 in Improved DDPM).
        # Without a factor of 1/1000, the VB term hurts the MSE term.
        vlb_loss *= self.num_timesteps / 1000.0

        # assume the learning target is the noise
        target = noise
        assert model_output.shape == target.shape == x_start.shape
        mse_loss = (target - model_output) ** 2
        # take the mean over all non-batch dimensions
        mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape))))
        loss = mse_loss + vlb_loss
        loss_dict = {
            f'losses/mse_loss': mse_loss.mean().item(),
            f'losses/vlb_loss': vlb_loss.mean().item(),
            f'train/log_var': model_log_var.mean().item()
        }
        return loss, loss_dict

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        '''
        Reverse diffusion process. Start from T, sample pure noise from isotropic gaussian, then use our model to
        gradually denoise it until t=0
        '''
        out = self.p_mean_variance(model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)['sample']
            imgs.append(img.cpu())

        return imgs

    def sample(self, model, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


def extract(a, t, x_shape):
    '''
    Extract the appropriate t-index for a batch of indices
    '''
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
