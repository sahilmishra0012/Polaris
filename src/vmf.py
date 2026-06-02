import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import SphericalLinear, SphericalTanh


def log_modified_bessel_first_kind(order, x):
    """Approximate log(I_order(x)) stably for vMF normalizers."""
    x = x.clamp_min(1e-8)
    order = torch.as_tensor(order, dtype=x.dtype, device=x.device)

    small_x = order * (torch.log(x) - math.log(2.0)) - \
        torch.lgamma(order + 1.0)

    z = torch.sqrt(x.pow(2) + order.pow(2))
    large_x = z - 0.5 * torch.log(2 * math.pi * z)
    if torch.any(order > 0):
        large_x = large_x + order * (torch.log(x) - torch.log(order + z))

    return torch.where(x < 1e-3, small_x, large_x)


def log_Z_d(kappa, d):
    """Compute the log normalizing constant for a vMF distribution."""
    kappa = kappa.clamp_min(1e-8)
    order = d / 2 - 1
    log_bessel_func = log_modified_bessel_first_kind(order, kappa)
    return order * torch.log(kappa) - (d / 2) * math.log(2 * math.pi) - log_bessel_func


def A_d(kappa, dim):
    """Compute the vMF mean resultant length approximation."""
    kappa = kappa.clamp_min(1e-8)
    half_minus = (dim - 1) / 2
    half_plus = (dim + 1) / 2
    return kappa / (half_minus + torch.sqrt(half_plus**2 + kappa.pow(2)))


def vmf_kl_divergence(mu1, kappa1, mu2, kappa2, dim):
    """Compute the KL divergence between vMF distributions."""
    log_Z_1 = log_Z_d(kappa1, dim)
    log_Z_2 = log_Z_d(kappa2, dim)

    A_d_1 = A_d(kappa1, dim)

    dot_product = torch.sum(mu1*mu2, dim=-1, keepdim=True)

    term1 = log_Z_1 - log_Z_2
    term2 = A_d_1 * (kappa1 - kappa2 * dot_product)

    return term1 + term2


class VMFRegularisation(nn.Module):
    """Regularization module for vMF-style spherical uncertainty parameters."""

    def __init__(self, args, embedding_dim, hidden_dim=64):
        """Initialize the VMFRegularisation object and its experiment state."""
        super(VMFRegularisation, self).__init__()

        self.embedding_dim = embedding_dim
        self.kappa_predictor = nn.Sequential(nn.Linear(
            embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Softplus())
        self.mu_predictor = nn.Sequential(SphericalLinear(
            embedding_dim, hidden_dim), SphericalTanh(), SphericalLinear(hidden_dim, embedding_dim))
        self.args = args

    def forward(self, p_emb, c_emb, n_emb, margin):
        """Run the forward pass and return loss terms."""
        if self.args.learn_mu == 0:
            mu_p = p_emb
            mu_c = c_emb
            mu_n = n_emb

            kappa_p = self.kappa_predictor(p_emb)
            kappa_c = self.kappa_predictor(c_emb)
            kappa_n = self.kappa_predictor(n_emb)
        elif self.args.learn_kappa == 0:
            mu_p, mu_c, mu_n = self.mu_predictor(
                p_emb), self.mu_predictor(c_emb), self.mu_predictor(n_emb)
            kappa_p = torch.full((p_emb.size(0), 1), 0.4,
                                 device=p_emb.device, dtype=p_emb.dtype)
            kappa_c = torch.full((c_emb.size(0), 1), 0.4,
                                 device=c_emb.device, dtype=c_emb.dtype)
            kappa_n = torch.full((n_emb.size(0), 1), 0.4,
                                 device=n_emb.device, dtype=n_emb.dtype)
        else:
            mu_p, mu_c, mu_n = self.mu_predictor(
                p_emb), self.mu_predictor(c_emb), self.mu_predictor(n_emb)
            kappa_p = self.kappa_predictor(p_emb)
            kappa_c = self.kappa_predictor(c_emb)
            kappa_n = self.kappa_predictor(n_emb)
        kl_pos = vmf_kl_divergence(
            mu_c, kappa_c, mu_p, kappa_p, self.embedding_dim)
        kl_neg = vmf_kl_divergence(
            mu_c, kappa_c, mu_n, kappa_n, self.embedding_dim)

        loss = F.relu(margin+kl_pos-kl_neg)

        return loss.mean(), mu_p, mu_c, mu_n
