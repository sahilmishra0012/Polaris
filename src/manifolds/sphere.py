import torch
from manifolds.base import Manifold


class Sphere(Manifold):
    def __init__(self):
        super(Sphere, self).__init__()

        self.name = 'Sphere'
        self.min_norm = 1e-6
        self.max_norm = 1000
        self.eps = 1e-6

    def proj_tan(self, x, v, c=0):
        dot_product = torch.sum(x*v, dim=-1, keepdim=True)

        return v-dot_product*x

    def expmap(self, x, v):
        norm_v = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=self.eps)

        exp_result = torch.cos(norm_v)*x+torch.sin(norm_v)*(v/norm_v)
        return exp_result

    def expmap_retracted(self, x, v, c=0):
        s = x+v
        exp_result = s/torch.norm(s, p=2, dim=1).unsqueeze(1)

        return exp_result

    def logmap(self, x, y, c=0):
        v = self.proj_tan(x, y)
        norm_v = torch.norm(
            v, p=2, dim=-1, keepdim=True).clamp_min(min=self.min_norm)

        dot_product = torch.sum(
            x*y, dim=-1, keepdim=True).clamp(min=-1.0+self.eps, max=1-self.eps)
        distance = torch.acos(dot_product)

        return distance*(v/norm_v)

    def parallel_transport(self, x, y, v):
        return self.proj_tan(y, v)
