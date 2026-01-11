import torch
import torch.nn.functional as F


class SVGD_Combined_Sphere:

    def __init__(self, kappa_repel=1.0, kappa_align=5.0, eps=1e-6):

        self.kappa_repel = kappa_repel
        self.kappa_align = kappa_align
        self.eps = eps

    def vmf_kernel(self, x, y):
        dot_products = torch.matmul(x, y.t())
        k = torch.exp(self.kappa_repel * dot_products)
        grad_k = self.kappa_repel * k.unsqueeze(-1) * y.unsqueeze(0)
        return k, grad_k

    def score_fn(self, x, mu):
        score_theta = torch.zeros_like(x)
        x_d = x[:, -1]
        score_theta[:, -1] = x_d / (1 - x_d.pow(2) + self.eps)

        score_align = self.kappa_align * mu

        return score_theta + score_align

    def __call__(self, x, mu):
        n_particles = x.size(0)
        k, grad_k_repulsion = self.vmf_kernel(x, x)

        repulsion = torch.sum(grad_k_repulsion, dim=1)

        score = self.score_fn(x, mu)
        drift = torch.matmul(k, score)

        svgd_grad = (drift + repulsion) / n_particles

        tangent_grad = svgd_grad - \
            (torch.sum(svgd_grad * x, dim=1, keepdim=True) * x)
        return tangent_grad


class SVGD_vMF_Sphere:
    def __init__(self, kappa=1):

        self.kappa = kappa

    def vmf_kernel(self, x, y):
        dot_products = torch.matmul(x, y.t())
        k = torch.exp(self.kappa * dot_products)

        grad_k = self.kappa * k.unsqueeze(-1) * y.unsqueeze(0)

        return k, grad_k

    def __call__(self, x):

        n_particles = x.size(0)
        k, grad_k = self.vmf_kernel(x, x)

        svgd_grad = torch.sum(grad_k, dim=1) / n_particles

        tangent_grad = svgd_grad - \
            (torch.sum(svgd_grad * x, dim=1, keepdim=True) * x)

        return tangent_grad


class SVGD_Uniform_Sphere:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def rbf_kernel(self, x):

        sq_dist = torch.cdist(x, x, p=2)**2
        h = sq_dist.median() / (2 * torch.log(torch.tensor(x.size(0), dtype=torch.float)))
        h = torch.sqrt(
            0.5 * h / torch.log(torch.tensor(x.size(0) + 1., dtype=torch.float)))

        k = torch.exp(-sq_dist / h**2 / 2)
        grad_k = -torch.einsum('ij, ik -> ijk', k, x) / (h**2)
        grad_k = grad_k + torch.einsum('ij, jk -> ijk', k, x) / (h**2)

        return k, grad_k

    def __call__(self, x):

        k, grad_k = self.rbf_kernel(x)

        svgd_grad = torch.sum(k.unsqueeze(-1) * grad_k,
                              dim=1) / x.size(0)

        tangent_grad = svgd_grad - \
            (torch.sum(svgd_grad * x, dim=1, keepdim=True) * x)

        return tangent_grad


class SVGD_IMQ_Sphere:
    def __init__(self, c=1.0, beta=-0.5):
        self.c = c
        self.beta = beta

    def imq_kernel(self, x, y):

        dot_products = torch.matmul(x, y.t())

        base = self.c + 2.0 - 2.0 * dot_products

        k = base.pow(self.beta)

        k_grad_coeff = -2.0 * self.beta * base.pow(self.beta - 1.0)

        grad_k = k_grad_coeff.unsqueeze(-1) * y.unsqueeze(0)

        return k, grad_k

    def __call__(self, x):
        n_particles = x.size(0)

        k, grad_k = self.imq_kernel(x, x)

        svgd_grad = torch.sum(grad_k, dim=1) / n_particles

        tangent_grad = svgd_grad - \
            (torch.sum(svgd_grad * x, dim=1, keepdim=True) * x)

        return tangent_grad


class SVGD_Periodic:

    def __init__(self, n_particles, bandwidth=None):
        self.n_particles = n_particles
        self.bandwidth = bandwidth

    def periodic_kernel(self, x):
        diff = torch.abs(x.unsqueeze(1) - x.unsqueeze(0))

        dist = torch.min(diff, 2 * torch.pi - diff)
        sq_dist = dist ** 2

        if self.bandwidth is None:
            h = torch.median(
                sq_dist) / (2 * torch.log(torch.tensor(self.n_particles + 1.0)))
            h = torch.sqrt(h + 1e-6)
        else:
            h = self.bandwidth

        k = torch.exp(-sq_dist / (2 * h**2 + 1e-6))

        x_expanded = x.unsqueeze(1)
        y_expanded = x.unsqueeze(0)
        delta = torch.remainder(
            x_expanded - y_expanded + torch.pi, 2 * torch.pi) - torch.pi

        grad_k = -(delta * k.unsqueeze(-1)) / (h**2 + 1e-6)

        return k, grad_k

    def __call__(self, x):
        k, grad_k = self.periodic_kernel(x)

        svgd_grad = torch.sum(grad_k, dim=1) / self.n_particles
        return svgd_grad
