import torch
import torch.nn.functional as F


class SVGD_Uniform_Theta_Sphere:

    def __init__(self, kappa=1.0, eps=1e-6):

        self.kappa = kappa
        self.eps = eps

    def vmf_kernel(self, x, y):
        dot_products = torch.matmul(x, y.t())
        k = torch.exp(self.kappa * dot_products)
        grad_k = self.kappa * k.unsqueeze(-1) * y.unsqueeze(0)
        return k, grad_k

    def score_fn(self, x):

        score = torch.zeros_like(x)
        x_d = x[:, -1]
        score[:, -1] = x_d / (1 - x_d.pow(2) + self.eps)
        return score

    def __call__(self, x):

        n_particles = x.size(0)

        k, grad_k_repulsion = self.vmf_kernel(x, x)

        repulsion = torch.sum(grad_k_repulsion, dim=1)

        score = self.score_fn(x)
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
    def __init__(self, n_particles, dim, bandwidth=1.0):
        self.n_particles = n_particles
        self.dim = dim
        self.bandwidth = bandwidth

    def rbf_kernel(self, x):

        sq_dist = torch.cdist(x, x, p=2)**2
        h = sq_dist.median() / (2 * torch.log(torch.tensor(self.n_particles, dtype=torch.float)))
        h = torch.sqrt(
            0.5 * h / torch.log(torch.tensor(self.n_particles + 1., dtype=torch.float)))

        k = torch.exp(-sq_dist / h**2 / 2)
        grad_k = -torch.einsum('ij, ik -> ijk', k, x) / (h**2)
        grad_k = grad_k + torch.einsum('ij, jk -> ijk', k, x) / (h**2)

        return k, grad_k

    def __call__(self, x):

        k, grad_k = self.rbf_kernel(x)

        svgd_grad = torch.sum(k.unsqueeze(-1) * grad_k,
                              dim=1) / self.n_particles

        tangent_grad = svgd_grad - \
            (torch.sum(svgd_grad * x, dim=1, keepdim=True) * x)

        return tangent_grad
