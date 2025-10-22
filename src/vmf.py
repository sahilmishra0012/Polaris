import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import SphericalLinear, SphericalTanh
try:
    from torch.special import i0
except ImportError:
    from torch import i0


def log_i0_stable(x):
    threshold = 20.0

    mask = x < threshold

    safe_x = torch.where(mask, x, torch.zeros_like(x))
    log_i0_safe = torch.log(i0(safe_x) + 1e-8)

    unsafe_x = torch.where(mask, torch.ones_like(x) * threshold, x)
    log_i0_approx = unsafe_x - 0.5 * torch.log(2 * math.pi * unsafe_x)

    return torch.where(mask, log_i0_safe, log_i0_approx)


def log_Z_d(kappa, d):
    log_bessel_func = log_i0_stable(kappa)
    return (((d/2)-1)*torch.log(kappa))-((0.5*d*math.log(2*math.pi))-log_bessel_func)


def A_d(kappa, dim):
    return (1-(dim-1))/(2*kappa+1e-8)


def vmf_kl_divergence(mu1, kappa1, mu2, kappa2, dim):
    log_Z_1 = log_Z_d(kappa1, dim)
    log_Z_2 = log_Z_d(kappa2, dim)

    A_d_1 = A_d(kappa1, dim)

    dot_product = torch.sum(mu1*mu2, dim=-1, keepdim=True)

    term1 = -kappa1*A_d_1
    term2 = kappa2*A_d_1*dot_product
    term3 = log_Z_2-log_Z_1

    return term1+term2+term3


class VMFRegularisation(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super(VMFRegularisation, self).__init__()

        self.embedding_dim = embedding_dim
        self.kappa_predictor = nn.Sequential(nn.Linear(
            embedding_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Softplus())
        self.mu_predictor = nn.Sequential(SphericalLinear(
            embedding_dim, hidden_dim), SphericalTanh(), SphericalLinear(hidden_dim, embedding_dim))

    def forward(self, p_emb, c_emb, n_emb, margin=1):
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

        return loss.mean()
