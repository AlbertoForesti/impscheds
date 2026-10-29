"""Analytic score of the binomial benchmark (no training needed)."""
import torch
import torch.nn as nn


class BinomialUniformOracle(nn.Module):
    """
    Analytic (ground-truth) score model for a 1D Binomial(n, p) categorical
    variable under the uniform-graph discrete diffusion forward process.

    Forward kernel (udm_schedules/noise.py, Scheduler.add_noise):
        q_t(x_t=a | x0) = exp(-sigma_bar) * [a == x0] + (1 - exp(-sigma_bar)) / S
    where S = num_vocabs. Marginalizing over x0 ~ Binomial(n, p) (n = num_vocabs-1):
        p_t(a) = exp(-sigma_bar) * Bin(a; n, p) + (1 - exp(-sigma_bar)) / S

    The true score ratio is s*(x_t, y) = p_t(y) / p_t(x_t); this module returns
    log(s*(x_t, ·)), matching the SEDD model output convention (score =
    output.exp(), see Scheduler.output_to_score), with the
    self-entry (y == x_t) zeroed (log(1) = 0, which is exact here anyway).

    No learnable parameters and no checkpoint.
    """

    def __init__(self, num_vocabs, p, eps=1e-12):
        super().__init__()
        self.num_vocabs = num_vocabs
        self.eps = eps
        n = num_vocabs - 1
        k = torch.arange(num_vocabs, dtype=torch.float64)
        log_binom_coeff = (
            torch.lgamma(torch.tensor(n + 1, dtype=torch.float64))
            - torch.lgamma(k + 1)
            - torch.lgamma(n - k + 1)
        )
        log_pmf = log_binom_coeff + k * torch.log(torch.tensor(p, dtype=torch.float64)) + (
            n - k
        ) * torch.log(torch.tensor(1 - p, dtype=torch.float64))
        true_pmf = log_pmf.exp().clamp_min(eps)
        true_pmf = true_pmf / true_pmf.sum()
        self.register_buffer("true_pmf", true_pmf.to(torch.float32))

    def forward(self, indices, sigma, **kwargs):
        # indices: (batch, length) long tensor of current noisy tokens
        # sigma: (batch,) float tensor -- this is sigma_bar(t), despite the
        # generic parameter name (callers use model(xt, sigma_bar)).
        true_pmf = self.true_pmf.to(indices.device, torch.float64)
        alpha = torch.exp(-sigma.to(torch.float64))  # (batch,)

        # p_t(a) for every vocab entry a, shape (batch, num_vocabs)
        p_t = alpha[:, None] * true_pmf[None, :] + (1 - alpha[:, None]) / self.num_vocabs

        length = indices.shape[1]
        p_t_expanded = p_t[:, None, :].expand(-1, length, -1)  # (batch, length, S)
        denom = torch.gather(p_t_expanded, -1, indices[..., None].to(torch.int64))  # (batch, length, 1)

        ratio = p_t_expanded / denom.clamp_min(self.eps)
        log_score = ratio.clamp_min(self.eps).log()
        log_score = torch.scatter(
            log_score, -1, indices[..., None], torch.zeros_like(log_score[..., :1])
        )
        return log_score.to(torch.float32)
