"""Datasets and evaluation metrics of the binomial and piano experiments."""
import numpy as np
import torch
import torch.nn.functional as F


############
# Binomial #
############
class BinomialDataset:
    """`length` i.i.d. components per sample, each drawn from the same
    Binomial(n, p) marginal with n = num_vocabs - 1. Only used to estimate
    the schedules; the score is analytic (oracle.BinomialUniformOracle)."""

    def __init__(self, num_samples=1024, num_vocabs=32, p=0.5, length=1):
        self.num_samples = num_samples
        self.num_vocabs = num_vocabs
        self.length = length
        n = num_vocabs - 1
        self.x0 = torch.distributions.Binomial(n, probs=p).sample((num_samples, length)).long()

    def __getitem__(self, index):
        return self.x0[index], torch.tensor(0)

    def __len__(self):
        return self.num_samples


def binomial_metric_fn(xt, num_vocabs=32, p=0.5, **kwargs):
    """KL(empirical || true) and 1-Wasserstein distance between the generated
    samples and the exact Binomial(num_vocabs-1, p) pmf. All positions share
    the same marginal, so all num_samples*length values are pooled into one
    empirical pmf."""
    n = num_vocabs - 1
    k = torch.arange(num_vocabs, dtype=torch.float64)
    true_pmf = torch.distributions.Binomial(n, probs=p).log_prob(k).exp()
    true_pmf = (true_pmf / true_pmf.sum()).numpy()

    samples = xt.cpu().numpy().reshape(-1)
    counts = np.bincount(samples, minlength=num_vocabs)[:num_vocabs].astype(np.float64)
    empirical_pmf = counts / counts.sum()

    true_cdf = np.cumsum(true_pmf)
    empirical_cdf = np.cumsum(empirical_pmf)
    wasserstein1 = float(np.sum(np.abs(true_cdf - empirical_cdf)))

    eps = 1e-12
    kl = float(np.sum(empirical_pmf * (np.log(empirical_pmf + eps) - np.log(true_pmf + eps))))

    return {"wasserstein1": wasserstein1, "kl": kl}


#########
# Piano #
#########
class PianoDataset:
    """Lakh pianoroll split as distributed with tauLDR: an (N, 256) array of
    tokens in [0, 129)."""

    def __init__(self, data_dir, split, num_samples=-1, device="cpu"):
        np_data = np.load(f"{data_dir}/{split}.npy")
        self.data = torch.from_numpy(np_data).to(device)
        self.num_samples = num_samples

    def __len__(self):
        if self.num_samples != -1 and self.num_samples < self.data.shape[0]:
            return self.num_samples
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], torch.tensor(0)


def piano_metric_fn(xt, ds, fix_length, num_vocabs, device="cpu", **kwargs):
    """Outlier score and Hellinger distance between the note histograms of
    each generated continuation and of the corresponding test continuation
    (the first `fix_length` tokens are the shared prompt)."""
    error_dict = {'outlier': [], 'hellinger': []}

    for i in range(len(ds)):
        fake_x0 = xt[i]
        true_x0, _ = ds[i]
        assert fake_x0.shape == true_x0.shape, (fake_x0.shape, true_x0.shape)
        assert (fake_x0[:fix_length].cpu() == true_x0[:fix_length].cpu()).all(), \
            "generated samples do not start with the conditioning tokens"
        fake_x0 = fake_x0[fix_length:].cpu()
        true_x0 = true_x0[fix_length:].cpu()
        fake_hist = F.one_hot(fake_x0, num_classes=num_vocabs).float().mean(dim=0).to(device)
        true_hist = F.one_hot(true_x0, num_classes=num_vocabs).float().mean(dim=0).to(device)

        error_dict['hellinger'].append(hellinger(fake_hist, true_hist).item())
        # proportion of generated notes that never occur in the true continuation
        error_dict['outlier'].append(fake_hist[true_hist == 0].sum().item())

    error_dict['hellinger'] = float(np.mean(error_dict['hellinger']))
    error_dict['outlier'] = float(np.mean(error_dict['outlier']))
    return error_dict


_SQRT2 = np.sqrt(2).item()
def hellinger(p, q):
    return torch.sqrt(torch.sum((p.sqrt() - q.sqrt()) ** 2)) / _SQRT2
