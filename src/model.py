import os
import pickle as pkl
import torch
import numpy as np
import sys
import torch.nn as nn
from utils import *
from layers import SphericalMLP
from transformers import BertModel, AutoModel
import torch.nn.functional as F
from vmf import VMFRegularisation
from manifolds.sphere import Sphere
from svgd import *
import open_clip


class PolarTaxo(nn.Module):
    """Polar orbital embedding model for taxonomy representation learning on spheres."""
    def __init__(self, args):
        """Initialize the PolarTaxo object and its experiment state."""
        super(PolarTaxo, self).__init__()

        self.args = args
        self.manifold = Sphere()
        self.input_dim = 768 if args.dataset != 'birds' else 1024

        if self.args.dataset != 'birds':
            self.pre_train_model = self.__load_pre_trained__()
        else:
            self.pre_train_model = self.args.pretrained_model.to('cuda')

        # Spherical
        self.vmf_regulariser = VMFRegularisation(args=self.args,
                                                 embedding_dim=self.args.embed_size, hidden_dim=self.args.hidden)
        self.child_sphere = SphericalMLP(
            input_dim=self.input_dim, hidden=self.args.hidden, output_dim=self.args.embed_size, bias=False)
        self.parent_sphere = SphericalMLP(
            input_dim=self.input_dim, hidden=self.args.hidden, output_dim=self.args.embed_size, bias=False)
        pole = torch.zeros(self.input_dim)
        pole[-1] = 1.0
        self.register_buffer("pole", pole.unsqueeze(0))

    def __load_pre_trained__(self):
        """Load the configured pretrained language model backbone."""
        if self.args.model == 'bert':
            model = BertModel.from_pretrained(
                '/home/models/bert-base-uncased')
        elif self.args.model == 'snowflake':
            model = AutoModel.from_pretrained(
                'Snowflake/snowflake-arctic-embed-m', add_pooling_layer=False)
        print("Model Loaded!")
        return model

    def get_image_cls(self, encode_inputs):
        """Encode image inputs with the pretrained multimodal backbone."""
        with torch.no_grad():
            encode_inputs = encode_inputs.view(encode_inputs.size(
                0), encode_inputs.size(2), encode_inputs.size(3), encode_inputs.size(4))
            image_embedding = self.pre_train_model.encode_image(encode_inputs)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

        return image_embedding

    def get_label_cls(self, encode_inputs):
        """Encode text label inputs with the pretrained multimodal backbone."""
        with torch.no_grad():
            encode_inputs = encode_inputs.view(
                encode_inputs.size(0), encode_inputs.size(2))
            text_embedding = self.pre_train_model.encode_text(encode_inputs)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

        return text_embedding

    def get_cls(self, encode_inputs):
        """Extract CLS-style text embeddings from the configured language model."""
        if self.args.model == 'snowflake':
            cls_embed = self.pre_train_model(
                **encode_inputs).last_hidden_state[:, 0]
        elif self.args.model == 'bert':

            outputs = self.pre_train_model(**encode_inputs)
            last_hidden_state = outputs.last_hidden_state
            cls_embed = last_hidden_state[:, 0, :]

        return cls_embed

    def multimodal_parent_projection(self, cls_embed):
        """Project candidate label embeddings onto the learned spherical space."""
        cls_embeddings = self.get_label_cls(cls_embed)
        v = self.manifold.proj_tan(self.pole, cls_embeddings)
        v_sphere = self.manifold.expmap_retracted(self.pole, v)

        e = self.parent_sphere(v_sphere)

        return e

    def multimodal_child_projection(self, cls_embed):
        """Project image query embeddings onto the learned spherical space."""
        cls_embeddings = self.get_image_cls(cls_embed)
        v = self.manifold.proj_tan(self.pole, cls_embeddings)
        v_sphere = self.manifold.expmap_retracted(self.pole, v)

        e = self.parent_sphere(v_sphere)

        return e

    def direct_projection_child(self, cls_embed):
        """Convert child embeddings directly into polar coordinates."""
        cls_embeddings = self.get_cls(cls_embed)

        psi, theta = self.to_polar(cls_embeddings)
        # e = self.parent_rect(e)

        # constrain within the rectangular manifold of 0 to pi
        # e = self.to_polar(e)

        return psi, theta

    def direct_projection_parent(self, cls_embed):
        """Convert parent embeddings directly into polar coordinates."""
        cls_embeddings = self.get_cls(cls_embed)

        psi, theta = self.to_polar(cls_embeddings)
        # e = self.parent_rect(e)

        return psi, theta

    def par_projection(self, cls_embed):

        """Project parent or candidate text embeddings onto the learned sphere."""
        cls_embeddings = self.get_cls(cls_embed)
        v = self.manifold.proj_tan(self.pole, cls_embeddings)
        v_sphere = self.manifold.expmap_retracted(self.pole, v)

        e = self.parent_sphere(v_sphere)

        return e

    def child_projection(self, cls_embed):

        """Project child or query text embeddings onto the learned sphere."""
        cls_embeddings = self.get_cls(cls_embed)
        v = self.manifold.proj_tan(self.pole, cls_embeddings)
        v_sphere = self.manifold.expmap_retracted(self.pole, v)

        e = self.parent_sphere(v_sphere)

        return e

    def to_polar(self, e):

        """Convert normalized Cartesian embeddings into polar angular coordinates."""
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

        return psi, theta.unsqueeze(1)

    def normalize_spherical_weights(self):
        """Renormalize spherical projection-layer weights after an optimizer step."""
        self.parent_sphere.l1.normalize_weights()
        self.parent_sphere.l2.normalize_weights()
        self.child_sphere.l1.normalize_weights()
        self.child_sphere.l2.normalize_weights()

    def angular_loss(self, psi1, psi2):

        """Compute aggregate latitudinal angular distance."""
        return torch.sum(torch.abs(psi1-psi2), dim=1)

    def longitudinal_loss(self, theta1, theta2):
        """Compute wrapped longitudinal angular distance."""
        diff = torch.abs(theta1-theta2)
        wrap_diff = 2*torch.pi-diff

        shortest_arc = torch.min(diff, wrap_diff)

        return torch.sum(shortest_arc, dim=1)

    def welsch_loss(self, d):

        """Compute the Welsch robust loss for angular distances."""
        w_loss = (self.args.c**2/2)*(1 -
                                     torch.exp(-(d**2/(2*self.args.c**2))))

        return w_loss

    def forward(self, step, encode_parent, encode_child, encode_negative):
        """Run the forward pass and return loss terms."""
        if self.args.implement_rectangular_opt is True:
            # Geometric
            parent_psi, parent_theta = self.direct_projection_parent(
                encode_parent)
            child_psi, child_theta = self.direct_projection_child(encode_child)
            negative_psi, negative_theta = self.direct_projection_parent(
                encode_negative)

            cp_loss = self.angular_loss(parent_psi, child_psi)
            cn_loss = self.angular_loss(negative_psi, child_psi)
            welsch_cp = torch.log(self.welsch_loss(cp_loss))
            welsch_cn = torch.log(self.welsch_loss(cn_loss))

            cp_long_loss = self.longitudinal_loss(parent_theta, child_theta)
            cn_long_loss = self.longitudinal_loss(negative_theta, child_theta)
            welsch_cp_long = torch.log(self.welsch_loss(cp_long_loss))
            welsch_cn_long = torch.log(self.welsch_loss(cn_long_loss))

            taxo_loss = F.relu(welsch_cp-welsch_cn+self.args.beta).mean() + \
                F.relu(welsch_cp_long-welsch_cn_long+self.args.beta).mean()

            # SVGD
            svgd_lat = SVGD_Uniform_Sphere()
            svgd_long = SVGD_Periodic(n_particles=parent_theta.size(0))
            parent_psi_grad, parent_theta_grad = svgd_lat(
                parent_psi), svgd_long(parent_theta)
            child_psi_grad, child_theta_grad = svgd_lat(
                child_psi), svgd_long(child_theta)
            negative_psi_grad, negative_theta_grad = svgd_lat(
                negative_psi), svgd_long(negative_theta)

            taxo_loss = F.relu(welsch_cp-welsch_cn+self.args.beta).mean()

            svgd_loss = parent_psi_grad.norm(p=2, dim=1).mean() + \
                child_psi_grad.norm(p=2, dim=1).mean() + \
                negative_psi_grad.norm(p=2, dim=1).mean()+parent_theta_grad.norm(p=2, dim=1).mean(
            )+child_theta_grad.norm(p=2, dim=1).mean()+negative_theta_grad.norm(p=2, dim=1).mean()

            final_loss = taxo_loss+self.args.svgd_weight*svgd_loss

        else:
            if self.args.dataset == 'birds':
                parent_sphere = self.multimodal_parent_projection(
                    encode_parent)
                child_sphere = self.multimodal_child_projection(encode_child)
                negative_sphere = self.multimodal_parent_projection(
                    encode_negative)
            else:
                parent_sphere = self.par_projection(encode_parent)
                child_sphere = self.child_projection(encode_child)
                negative_sphere = self.par_projection(encode_negative)

            dot_cp = torch.sum(parent_sphere*child_sphere, dim=1)
            dot_cn = torch.sum(negative_sphere*child_sphere, dim=1)

            dot_cp = torch.clamp(
                dot_cp, -1.0 + self.args.eps, 1.0 - self.args.eps)
            dot_cn = torch.clamp(
                dot_cn, -1.0 + self.args.eps, 1.0 - self.args.eps)

            ang_distcp = torch.acos(dot_cp)
            ang_distcn = torch.acos(dot_cn)

            welsch_cp = torch.log(self.welsch_loss(ang_distcp))
            welsch_cn = torch.log(self.welsch_loss(ang_distcn))

            loss_vmf, mu_p, mu_c, mu_n = self.vmf_regulariser(
                parent_sphere, child_sphere, negative_sphere, self.args.vmf_margin)

            if self.args.detach_svgd == 0:
                if self.args.kernel_setting == 'radial':
                    svgd_combined = SVGD_Uniform_Sphere()

                    parent_svgd_grad = svgd_combined(parent_sphere)
                    child_svgd_grad = svgd_combined(child_sphere)
                    negative_svgd_grad = svgd_combined(negative_sphere)

                elif self.args.kernel_setting == 'vmf':
                    svgd_combined = SVGD_vMF_Sphere()

                    parent_svgd_grad = svgd_combined(parent_sphere)
                    child_svgd_grad = svgd_combined(child_sphere)
                    negative_svgd_grad = svgd_combined(negative_sphere)

                elif self.args.kernel_setting == 'imq':

                    svgd_combined = SVGD_IMQ_Sphere()

                    parent_svgd_grad = svgd_combined(parent_sphere)
                    child_svgd_grad = svgd_combined(child_sphere)
                    negative_svgd_grad = svgd_combined(negative_sphere)

                elif self.args.kernel_setting == 'vmf_theta':
                    k_repel = self.args.kappa_repel
                    k_align = self.args.kappa_align
                    svgd_combined = SVGD_Combined_Sphere(
                        kappa_align=k_align, kappa_repel=k_repel)

                if self.args.experiment_setting == 'constant_svgd':
                    parent_svgd_grad = svgd_combined(
                        parent_sphere, parent_sphere)
                    child_svgd_grad = svgd_combined(child_sphere, child_sphere)
                    negative_svgd_grad = svgd_combined(
                        negative_sphere, negative_sphere)
                else:
                    parent_svgd_grad = svgd_combined(parent_sphere, mu_p)
                    child_svgd_grad = svgd_combined(child_sphere, mu_c)
                    negative_svgd_grad = svgd_combined(negative_sphere, mu_n)

                taxo_loss = self.args.geometric_weight * \
                    F.relu(welsch_cp-welsch_cn+self.args.beta).mean() + \
                    (1-self.args.geometric_weight)*loss_vmf
                svgd_loss = parent_svgd_grad.norm(p=2, dim=1).mean() + \
                    child_svgd_grad.norm(p=2, dim=1).mean() + \
                    negative_svgd_grad.norm(p=2, dim=1).mean()
                final_loss = taxo_loss+(self.args.svgd_weight*svgd_loss)
            else:
                taxo_loss = self.args.geometric_weight * \
                    F.relu(welsch_cp-welsch_cn+self.args.beta).mean() + \
                    (1-self.args.geometric_weight)*loss_vmf
                final_loss = taxo_loss
                svgd_loss = torch.tensor(0)

        return final_loss, taxo_loss, svgd_loss
