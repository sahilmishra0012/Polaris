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

        # self.par_projection_h = MLP(
        #     input_dim=768, hidden=self.args.hidden, output_dim=1)
        self.par_projection_theta = MLP(
            input_dim=767, hidden=self.args.hidden, output_dim=self.args.embed_size)

        # self.child_projection_h = MLP(
        #     input_dim=1, hidden=self.args.hidden, output_dim=1)
        self.child_projection_theta = MLP(
            input_dim=767, hidden=self.args.hidden, output_dim=self.args.embed_size)

        # self.dropout = nn.Dropout(self.args.dropout)

        # self.child_sphere_projection = SphericalProjectionHead(
        #     self.args, 768, self.args.embed_size+2, self.args.embed_size)
        # self.parent_sphere_projection = SphericalProjectionHead(
        #     self.args, 768, self.args.embed_size+2, self.args.embed_size
        # )
        # self.child_sphere_projection = CartesianToPolarConverter()
        # self.parent_sphere_projection = CartesianToPolarConverter()

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

    def positive_last_dim_bert(self, x):
        x_pos = x.clone()

        last = x[..., -1]
        last_pos = last.abs().detach()+(last-last.detach())
        x_pos[..., -1] = last_pos

        return x_pos

    def par_projection(self, cls_embed):

        cls_embeddings = self.get_cls(cls_embed)
        v = self.manifold.proj_tan(self.pole, cls_embeddings)
        v_sphere = self.manifold.expmap_retracted(self.pole, v)

        e = self.parent_sphere(v_sphere)

        return e

        # We are on the surface of the sphere

        # theta = self.par_projection_h(self.get_cls(cls_embed))
        # psi = self.par_projection_theta(self.get_cls(cls_embed))

        # theta = 2*torch.pi*F.sigmoid(theta)
        # psi = torch.pi*F.sigmoid(psi)
        # theta, psi = self.parent_sphere_projection(self.get_cls(cls_embed))

        # cls_embeddings = self.get_cls(cls_embed)
        # cls_embeddings = self.positive_last_dim_bert(cls_embeddings)
        # psi_bert = self.parent_sphere_projection(cls_embeddings)
        # psi_proj = self.par_projection_theta(psi_bert)
        # psi_proj = F.sigmoid(psi_proj)*torch.pi

        # return psi_proj

    def child_projection(self, cls_embed):

        cls_embeddings = self.get_cls(cls_embed)
        v = self.manifold.proj_tan(self.pole, cls_embeddings)
        v_sphere = self.manifold.expmap_retracted(self.pole, v)

        e = self.parent_sphere(v_sphere)

        return e

        # theta = self.child_projection_h(self.get_cls(cls_embed))
        # psi = self.child_projection_theta(self.get_cls(cls_embed))

        # theta = 2*torch.pi*F.sigmoid(theta)
        # psi = torch.pi*F.sigmoid(psi)

        # theta = 2*torch.pi*F.sigmoid(theta)
        # psi = torch.pi*F.sigmoid(psi)
        # cls_embeddings = self.get_cls(cls_embed)
        # cls_embeddings = self.positive_last_dim_bert(cls_embeddings)
        # psi = self.child_sphere_projection(cls_embeddings)
        # psi_proj = self.child_projection_theta(psi)
        # psi_proj = F.sigmoid(psi_proj)*torch.pi

        # return psi_proj
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

    def longitude_distance(self, angles1, angles2):
        direct = torch.sum(torch.abs(angles1 - angles2),
                           dim=1, keepdim=True)   # [B,1]
        wrapped = torch.sum(
            torch.abs(2*torch.pi - angles1 - angles2), dim=1, keepdim=True)

        return torch.min(direct, wrapped)

    def latitude_distance(self, angles1, angles2):
        return torch.sum(torch.abs(angles1-angles2), dim=1, keepdim=True)

    def welsch_loss(self, d):

        w_loss = (self.args.c**2/2)*(1 -
                                     torch.exp(-(d**2/(2*self.args.c**2))))

        return w_loss

    def forward(self, step, encode_parent, encode_child, encode_negative):

        parent_sphere = self.par_projection(encode_parent)
        child_sphere = self.child_projection(encode_child)
        negative_sphere = self.par_projection(encode_negative)

        # # move these to the spherical coordinate system (theta,psi,psi1....)
        # parent_angles = self.to_polar(parent_sphere)
        # child_angles = self.to_polar(child_sphere)
        # negative_angles = self.to_polar(negative_sphere)

        # # Note that psi is between 0 to 2pi(longitudinal) and psis is 0 to pi.
        # parent_psi = parent_angles[:, -1].view(parent_angles.size(0), 1)
        # child_psi = child_angles[:, -1].view(parent_angles.size(0), 1)
        # negative_psi = negative_angles[:, -1].view(parent_angles.size(0), 1)

        # parent_thetas = parent_angles[:, :-1]
        # child_thetas = child_angles[:, :-1]
        # negative_thetas = negative_angles[:, :-1]

        # pos_long_distance = self.longitude_distance(parent_psi, child_psi)
        # neg_long_distanc = self.longitude_distance(negative_psi, child_psi)
        # pos_lat_distance = self.latitude_distance(parent_thetas, child_thetas)
        # neg_lat_distance = self.latitude_distance(
        #     negative_thetas, child_thetas)

        # # For longitude
        # welsch_long_cp = torch.log(self.welsch_loss(pos_long_distance))
        # welsch_long_cn = torch.log(self.welsch_loss(neg_long_distanc))
        # loss_longtiude = F.relu(
        #     welsch_long_cp-welsch_long_cn+self.args.beta).mean()

        # # for latitude
        # welsch_lat_cp = torch.log(self.welsch_loss(pos_lat_distance))
        # welsch_lat_cn = torch.log(self.welsch_loss(neg_lat_distance))
        # loss_latitude = F.relu(
        #     welsch_lat_cp-welsch_lat_cn+self.args.beta).mean()

        dot_cp = torch.sum(parent_sphere*child_sphere, dim=1)
        dot_cn = torch.sum(negative_sphere*child_sphere, dim=1)

        dot_cp = torch.clamp(dot_cp, -1.0 + self.args.eps, 1.0 - self.args.eps)
        dot_cn = torch.clamp(dot_cn, -1.0 + self.args.eps, 1.0 - self.args.eps)

        ang_distcp = torch.acos(dot_cp)
        ang_distcn = torch.acos(dot_cn)

        welsch_cp = torch.log(self.welsch_loss(ang_distcp))
        welsch_cn = torch.log(self.welsch_loss(ang_distcn))

        # svgd_loss = self.svgd_loss(
        #     parent_sphere, child_sphere, negative_sphere)

        loss = F.relu(welsch_cp-welsch_cn+self.args.beta).mean()+self.vmf_regulariser(
            parent_sphere, child_sphere, negative_sphere, self.args.vmf_margin)

        return loss

        # latitude_distance = F.relu(self.args.beta+self.welsch_loss(self.latitude_distance(
        #     parent_theta, child_theta))-self.welsch_loss(self.latitude_distance(negative_theta, child_theta)))
        # longitude_distance = F.relu(self.args.beta+self.welsch_loss(self.longitude_distance(
        #     parent_psi, child_psi))-self.welsch_loss(self.longitude_distance(negative_psi, child_psi)))

        # angle_loss = (self.args.lat_weight*latitude_distance +
        #               self.args.long_weight*longitude_distance).mean()

        # # margin_sq = self.args.beta**2
        # if (step+1) % 25 == 0:
        #     self.plot_and_save_distributions(child_theta, child_psi, step)

        # all_thetas = torch.cat(
        #     [parent_theta, child_theta, negative_theta], dim=0)

        # all_psis = torch.cat([parent_psi, child_psi, negative_psi], dim=0)
        # svgd_loss = self.compute_svgd_uniformity(all_thetas, all_psis)

        # loss = angle_loss+0.1*svgd_loss

        # return loss
        # parent_psi = self.par_projection(encode_parent)
        # child_psi = self.child_projection(encode_child)
        # negative_psi = self.par_projection(encode_negative)

        # latitude_distance = F.relu(self.args.beta+self.welsch_loss(self.latitude_distance(
        #     parent_psi, child_psi))-self.welsch_loss(self.latitude_distance(negative_psi, child_psi)))

        # if (step+1) % 25 == 0:
        #     self.plot_psi_distributions(
        #         step, parent_psi, child_psi, negative_psi)

        # psis = torch.cat([parent_psi, child_psi, negative_psi], dim=0)
        # svg_loss = self.compute_svgd_uniformity(psis)

        # loss = (0.9*latitude_distance+(0.1*svg_loss)).mean()

        # return loss
