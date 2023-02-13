import torch

from tqdm import tqdm
from diffusion.forward_diffusion import extract


@torch.no_grad()
def p_sample(model, x, t, t_index, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance):
    '''
    Reverse diffusion process. Start from T, sample pure noise from isotropic gaussian, then use our model to
    gradually denoise it until t=0
    '''
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Eq. 11 in the paper
    # use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise



# algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape, timesteps, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b, ), i, device=device, dtype=torch.long), i,
                       sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance)
        imgs.append(img.cpu().numpy())

    return imgs


def sample(model, image_size, timesteps, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, batch_size=16, channels=3):
    return p_sample_loop(model,
                         shape=(batch_size, channels, image_size, image_size),
                         timesteps=timesteps,
                         sqrt_recip_alphas=sqrt_recip_alphas,
                         sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                         posterior_variance=posterior_variance)