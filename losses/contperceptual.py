# https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/losses/contperceptual.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoders.discriminator import NLayerDiscriminator, weights_init
from autoencoders.policy.hypernet import ModelEncoder


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class LPIPS(nn.Module):
    def __init__(self, behavior_predictor: ModelEncoder, spatial: bool = False):
        '''
        Use perceptual loss to guide the latents to accurately encode relevant behavior information
        :param behavior_predictor: policy params -> measure predictor
        based off of https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
        '''
        super().__init__()
        self.behavior_predictor = behavior_predictor
        self.behavior_predictor.eval()
        self.spatial = spatial

    def forward(self,
                gt_policy_weights: dict[torch.Tensor],
                rec_policy_weights: dict[torch.Tensor],
                normalize: bool = False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            for (n1, w1), (n2, w2) in zip(gt_policy_weights.items(), rec_policy_weights.items()):
                gt_policy_weights[n1] = 2 * w1 - 1
                rec_policy_weights[n2] = 2 * w2 - 1

        # TODO: scaling layer. I think we don't need this b/c the original code scales in order to feed into the
        # TODO: but we don't do this or need to for the model encoder

        gt_preds = self.behavior_predictor(gt_policy_weights, get_intermediate_features=True)
        rec_preds = self.behavior_predictor(rec_policy_weights, get_intermediate_features=True)
        gt_features, rec_features, diffs = {}, {}, {}

        num_features = len(self.behavior_predictor.cnns) + 1  # all cnns + 2nd to last dense layer
        for kk in range(num_features):
            gt_features[kk], rec_features[kk] = normalize_tensor(gt_preds[kk]), normalize_tensor(rec_preds[kk])
            diffs[kk] = (gt_features[kk] - rec_features[kk])**2

        if self.spatial:
            # TODO: not sure what this is for or if we need it for policy params
            pass
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(num_features)]

        val = 0
        for l in range(num_features):
            val += res[l]

        return val.view(-1)


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 pixelloss_weight=1.0,
                 disc_num_layers=2,
                 disc_in_channels=1,
                 disc_factor=1.0,
                 disc_weight=1.0,
                 use_actnorm=False):
        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm).apply(weights_init)
        self.discriminator.to(torch.device('cuda'))
        self.disc_iter_start = disc_start
        self.disc_loss = hinge_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, global_step, optimizer_idx, last_layer=None):
        # standard mse + kl regularization loss
        rec_loss = torch.pow(inputs - reconstructions, 2.)
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)
            if self.disc_factor > 0:
                try:
                    # d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                    d_weight = self.discriminator_weight
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_iter_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            return loss
        elif optimizer_idx == 1:
            # discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.disc_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            return d_loss

