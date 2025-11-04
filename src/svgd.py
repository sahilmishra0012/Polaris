import torch


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
