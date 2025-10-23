import os
import pickle as pkl
import torch
import numpy as np
import sys
import torch.nn as nn
from utils import *
from layers import MLP, SphericalMLP
from transformers import BertModel, AutoModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
from vmf import VMFRegularisation
from manifolds.sphere import Sphere


class PolarTaxo(nn.Module):
    def __init__(self, args):
        super(PolarTaxo, self).__init__()

        self.args = args
        self.manifold = Sphere()

        self.pre_train_model = self.__load_pre_trained__()

        # Spherical
        self.vmf_regulariser = VMFRegularisation(
            embedding_dim=self.args.embed_size, hidden_dim=self.args.hidden)
        self.child_sphere = SphericalMLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, bias=False)
        self.parent_sphere = SphericalMLP(
            input_dim=768, hidden=self.args.hidden, output_dim=self.args.embed_size, bias=False)
        pole = torch.zeros(768)
        pole[-1] = 1.0
        self.register_buffer("pole", pole.unsqueeze(0))

    def __load_pre_trained__(self):
        if self.args.model == 'bert':
            model = BertModel.from_pretrained(
                '/home/models/bert-base-uncased')
        elif self.args.model == 'snowflake':
            model = AutoModel.from_pretrained(
                'Snowflake/snowflake-arctic-embed-m', add_pooling_layer=False)
        print("Model Loaded!")
        return model

    def get_cls(self, encode_inputs):
        if self.args.model == 'snowflake':
            cls_embed = self.pre_train_model(
                **encode_inputs).last_hidden_state[:, 0]
        elif self.args.model == 'bert':
            outputs = self.pre_train_model(**encode_inputs)
            last_hidden_state = outputs.last_hidden_state

            cls_embed = last_hidden_state[:, 0, :]

        return cls_embed

    def par_projection(self, cls_embed):

        cls_embeddings = self.get_cls(cls_embed)
        v = self.manifold.proj_tan(self.pole, cls_embeddings)
        v_sphere = self.manifold.expmap_retracted(self.pole, v)

        e = self.parent_sphere(v_sphere)

        return e

    def child_projection(self, cls_embed):

        cls_embeddings = self.get_cls(cls_embed)
        v = self.manifold.proj_tan(self.pole, cls_embeddings)
        v_sphere = self.manifold.expmap_retracted(self.pole, v)

        e = self.parent_sphere(v_sphere)

        return e

    def to_polar(self, e):

        batch_size, d = e.shape
        if d < 2:
            raise ValueError(
                "Input Cartesian vector dimension must be at least 2 for conversion.")

        eps = 1e-8

        e_norm = e / (torch.norm(e, p=2, dim=1, keepdim=True) + eps)

        e_sq = e_norm.pow(2)

        cum_sq_from_back = torch.cumsum(torch.flip(e_sq, dims=[1]), dim=1)
        cum_sq_from_back = torch.flip(cum_sq_from_back, dims=[1])

        num_psi = d - 2
        psi = torch.zeros(batch_size, num_psi, device=e.device)
        for i in range(num_psi):
            numerator = e_norm[:, i]
            denominator = torch.sqrt(cum_sq_from_back[:, i] + eps)
            ratio = numerator / denominator

            clamped_ratio = torch.clamp(ratio, -1.0 + eps, 1.0 - eps)
            psi[:, i] = torch.acos(clamped_ratio)

        e_d_minus_1 = e_norm[:, d - 2]
        e_d = e_norm[:, d - 1]

        theta_denom = torch.sqrt(e_d_minus_1**2 + e_d**2 + eps)
        theta_ratio = e_d_minus_1 / theta_denom

        clamped_theta_ratio = torch.clamp(theta_ratio, -1.0 + eps, 1.0 - eps)
        theta_base = torch.acos(clamped_theta_ratio)
        theta = torch.where(e_d < 0, 2 * math.pi - theta_base, theta_base)

        return torch.cat([psi, theta.unsqueeze(1)], dim=1)

    def normalize_spherical_weights(self):
        self.parent_sphere.l1.normalize_weights()
        self.parent_sphere.l2.normalize_weights()
        self.child_sphere.l1.normalize_weights()
        self.child_sphere.l2.normalize_weights()

    def welsch_loss(self, d):

        w_loss = (self.args.c**2/2)*(1 -
                                     torch.exp(-(d**2/(2*self.args.c**2))))

        return w_loss

    def forward(self, step, encode_parent, encode_child, encode_negative):

        parent_sphere = self.par_projection(encode_parent)
        child_sphere = self.child_projection(encode_child)
        negative_sphere = self.par_projection(encode_negative)

        dot_cp = torch.sum(parent_sphere*child_sphere, dim=1)
        dot_cn = torch.sum(negative_sphere*child_sphere, dim=1)

        dot_cp = torch.clamp(dot_cp, -1.0 + self.args.eps, 1.0 - self.args.eps)
        dot_cn = torch.clamp(dot_cn, -1.0 + self.args.eps, 1.0 - self.args.eps)

        ang_distcp = torch.acos(dot_cp)
        ang_distcn = torch.acos(dot_cn)

        welsch_cp = torch.log(self.welsch_loss(ang_distcp))
        welsch_cn = torch.log(self.welsch_loss(ang_distcn))

        loss = self.args.geometric_weight*F.relu(welsch_cp-welsch_cn+self.args.beta).mean()+(1-self.args.geometric_weight)*self.vmf_regulariser(
            parent_sphere, child_sphere, negative_sphere, self.args.vmf_margin)

        return loss
