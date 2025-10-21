import torch
import torch.nn as nn
from manifolds.sphere import Sphere
import numpy as np
import torch.nn.functional as F
import math


class CartesianToPolarConverter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, e: torch.Tensor) -> torch.Tensor:

        batch_size, d = e.shape
        if d < 2:
            raise ValueError(
                "Input Cartesian vector dimension must be at least 2 for conversion.")

        e = e / (torch.norm(e, p=2, dim=1, keepdim=True) + 1e-9)

        e_sq = e.pow(2)

        cum_sq_from_back = torch.cumsum(torch.fliplr(e_sq), dim=1)
        cum_sq_from_back = torch.fliplr(cum_sq_from_back)

        num_psi = d - 2
        psi = torch.zeros(batch_size, num_psi, device=e.device)
        for i in range(num_psi):
            numerator = e[:, i]

            denominator = torch.sqrt(cum_sq_from_back[:, i])

            ratio = numerator / (denominator + 1e-9)

            clamped_ratio = torch.clamp(ratio, -1.0, 1.0)
            psi[:, i] = torch.acos(clamped_ratio)

        e_d_minus_1 = e[:, d-2]
        e_d = e[:, d-1]

        theta_denom = torch.sqrt(e_d_minus_1**2 + e_d**2)
        theta_ratio = e_d_minus_1 / (theta_denom + 1e-9)

        clamped_theta_ratio = torch.clamp(theta_ratio, -1.0, 1.0)
        theta_base = torch.acos(clamped_theta_ratio)

        theta = torch.where(e_d < 0, 2 * math.pi - theta_base, theta_base)

        return torch.cat([psi, theta.unsqueeze(1)], dim=1)


class SphericalLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight.is_manifold = True
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -1, 1)

        with torch.no_grad():
            self.normalize_weights()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def normalize_weights(self):
        self.weight.data = F.normalize(self.weight.data, dim=1, p=2)

    def forward(self, x):

        linear_transformation = F.linear(x, self.weight)
        spherical_projection = F.normalize(linear_transformation, p=2, dim=1)

        if self.bias is not None:
            spherical_projection = spherical_projection+self.bias
            spherical_projection = F.normalize(
                spherical_projection, dim=1, p=2)

        return spherical_projection


class SphericalReLU(nn.Module):
    def __init__(self):
        super(SphericalReLU, self).__init__()
        self.manifold = Sphere()

    def forward(self, x, pole):
        x = self.manifold.proj_tan(pole, x)
        x = torch.relu(x)
        x = self.manifold.expmap(pole, x)

        return x


class SphericalMLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, bias=False):
        super(SphericalMLP, self).__init__()
        self.l1 = SphericalLinear(input_dim, hidden, bias)
        self.l2 = SphericalLinear(hidden, output_dim, bias)
        # self.relu = SphericalReLU()

        # pole = torch.zeros(hidden)
        # pole[-1] = 1.0
        # self.register_buffer("pole", pole.unsqueeze(0))

    def forward(self, x):
        x = self.l1(x)
        x = F.normalize(torch.tanh(x), p=2, dim=-1)
        # x = self.relu(x, self.pole)
        x = self.l2(x)

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class SphericalProjectionHeadPsiOnly(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.args = args

    def differentiable_cartesian_to_psi(self, cartesian_vectors: torch.Tensor) -> torch.Tensor:

        norm = torch.norm(cartesian_vectors, p=2, dim=1, keepdim=True)
        unit_vectors = cartesian_vectors / (norm + 1e-9)

        batch_size, n_dim = unit_vectors.shape
        num_angles = n_dim - 1

        psi_angles = torch.zeros(
            batch_size, num_angles, device=unit_vectors.device)

        cumulative_sum_sq = torch.zeros(batch_size, device=unit_vectors.device)

        for i in range(n_dim - 1, 0, -1):
            numerator = unit_vectors[:, i-1]

            cumulative_sum_sq = cumulative_sum_sq + unit_vectors[:, i]**2

            denominator = torch.sqrt(cumulative_sum_sq + numerator**2)
            denominator = denominator + 1e-9

            ratio = numerator / denominator

            clamped_ratio = torch.clamp(ratio, -1.0, 1.0)

            psi_angles[:, i-1] = torch.acos(clamped_ratio)

        return psi_angles

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        psi = self.differentiable_cartesian_to_psi(x)
        return psi


class SphericalProjectionHead(nn.Module):
    def __init__(self, args, input_dim, feature_dim, e_dim):

        super().__init__()

        self.args = args

    def differentiable_cartesian_to_polar(self, cartesian_vectors):

        norm = torch.norm(cartesian_vectors, p=2, dim=1, keepdim=True)
        unit_vectors = cartesian_vectors / (norm + 1e-9)

        n_dim = unit_vectors.shape[1]
        angles = torch.zeros(
            unit_vectors.shape[0], n_dim - 1, device=unit_vectors.device)

        cumulative_sum_sq = torch.zeros_like(unit_vectors[:, 0])

        for i in range(n_dim - 2, 0, -1):
            x_i = unit_vectors[:, i]
            cumulative_sum_sq = cumulative_sum_sq + unit_vectors[:, i+1]**2
            denominator = torch.sqrt(cumulative_sum_sq + x_i**2)

            denominator = denominator + 1e-9

            ratio = x_i / denominator

            clamped_ratio = torch.clamp(ratio, -1.0, 1.0)
            angles[:, i] = torch.acos(clamped_ratio)

        x_0 = unit_vectors[:, 0]
        angles[:, 0] = torch.acos(torch.clamp(x_0, -1.0, 1.0))

        x_last = unit_vectors[:, -1]
        x_second_last = unit_vectors[:, -2]

        theta_denom = torch.sqrt(x_second_last**2 + x_last**2) + 1e-9
        theta_ratio = x_second_last / theta_denom
        theta = torch.acos(torch.clamp(theta_ratio, -1.0, 1.0))

        angles[:, -1] = torch.where(x_last < 0, 2 * math.pi - theta, theta)

        final_theta = angles[:, -1].unsqueeze(1)
        final_psi = angles[:, :-1]

        return final_theta, final_psi

    def forward(self, x):
        cartesian_unit_vectors = x / \
            torch.norm(x, p=2, dim=1, keepdim=True)
        print("Norm: ", torch.norm(x, p=2, dim=1, keepdim=True))
        theta, psi = self.differentiable_cartesian_to_polar(
            cartesian_unit_vectors)

        return
