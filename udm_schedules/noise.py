# Uniform-graph discrete diffusion (SEDD parameterization) used with the
# analytic binomial score. Adapted from the Jump Your Steps code base
# (https://github.com/enkeejunior1/jump-your-steps), itself based on SEDD
# (https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class Scheduler(nn.Module):
    """Uniform transition kernel with a loglinear or linear noise schedule.

    Both schedules are matched at the endpoints, alpha(t) = exp(-sigma_bar(t))
    goes from 1 to eps over t in [0, 1]:
        loglinear: sigma_bar(t) = -log(1 - (1 - eps) t)   (accelerating rate)
        linear:    sigma_bar(t) = -log(eps) t             (constant rate)
    """
    def __init__(self, args):
        super().__init__()
        self.length = args.length
        self.eps = args.eps
        self.num_vocabs = args.num_vocabs

        if args.noise_schedule == "loglinear":
            self.sigma_bar = lambda t: -torch.log1p(-(1 - self.eps) * t)
            self.sigma = lambda t: (1 - self.eps) / (1 - (1 - self.eps) * t)
        elif args.noise_schedule == "linear":
            neg_log_eps = -math.log(self.eps)
            self.sigma_bar = lambda t: neg_log_eps * t
            self.sigma = lambda t: neg_log_eps * torch.ones_like(t)
        else:
            raise ValueError(f"unknown noise schedule: {args.noise_schedule}")

    def add_noise(
        self, samples: torch.LongTensor, t: Union[int, torch.LongTensor], generator=None,
    ):
        '''x0 -> xt'''
        sigma_bar = self.sigma_bar(t)
        perturb_prob = 1 - (-sigma_bar).exp()
        move_indices = torch.rand(*samples.shape, device=samples.device, generator=generator) < perturb_prob[:, None]
        return torch.where(move_indices, torch.randint_like(samples, self.num_vocabs), samples)

    def output_to_score(self, output):
        return output.exp()

    def sample_latent(self, num_samples):
        return torch.randint(0, self.num_vocabs, (num_samples, self.length)).long()


class SchedulerOutput:
    def __init__(self, xt, xt_prob=None, rev_rate=None):
        self.xt = xt
        self.xt_prob = xt_prob
        self.rev_rate = rev_rate


class EulerScheduler(Scheduler):
    def Q_tok(self, i):
        '''Q_tok = Q[i, :] (Eq.16 from SEDD paper)'''
        edge = torch.ones(*i.shape, self.num_vocabs, device=i.device) / self.num_vocabs
        edge = edge.scatter(-1, i[..., None], - (self.num_vocabs - 1) / self.num_vocabs)
        return edge

    def Q_tilde(self, xt, score):
        normalized_rate = self.Q_tok(xt) * score
        # ref: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/graph_lib.py
        # to ensure that maintain the rate matrix property (sum_j R_ij = 0)
        normalized_rate.scatter_(-1, xt[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, xt[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def step(self, output, xt, t, step_size, rev_rate=None, generator=None, if_last=False):
        if rev_rate is None:
            sigma = self.sigma(t)
            score = self.output_to_score(output)
            score = torch.where(torch.isnan(score), 1e12 * torch.sign(score), score)
            score = torch.where(torch.isinf(score), 1e12 * torch.sign(score), score)
            rev_rate = sigma[..., None, None] * self.Q_tilde(xt, score)
        identity = F.one_hot(xt, num_classes=self.num_vocabs).to(rev_rate)
        xt_prob = identity + step_size * rev_rate
        xt = sample_categorical(xt_prob, generator=generator)
        return SchedulerOutput(xt, xt_prob=xt_prob, rev_rate=rev_rate)


def sample_categorical(categorical_probs, eps=1e-6, generator=None):
    '''use gumbel-max trick, but given probability'''
    if generator is None:
        gumbel_noise = torch.rand_like(categorical_probs)
    else:
        gumbel_noise = torch.rand(categorical_probs.shape, generator=generator, device=generator.device).to(categorical_probs)
    gumbel_noise = (eps - torch.log(eps + (1 - eps) * gumbel_noise))
    return torch.argmax(categorical_probs / gumbel_noise, dim=-1)
