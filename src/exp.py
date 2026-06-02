import os
import time
import numpy as np
import pickle as pkl
import torch
import sys
import json
from tqdm import tqdm
from torch import optim
from transformers import BertTokenizer, AutoTokenizer, DistilBertTokenizer
from utils import *
from data import *
from model import PolarTaxo
import matplotlib.pyplot as plt
from optimizer import RiemannianAdam, PolarEmbeddingsOptimizer
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import gc
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from hooks import GradientLogger
import csv
import wandb
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

os.environ["WANDB_MODE"] = "online"


class Experiments(object):

    """Training, evaluation, and analysis driver for PolarTaxo experiments."""
    def __init__(self, args):
        """Initialize the Experiments object and its experiment state."""
        super(Experiments, self).__init__()

        self.args = args
        self.tokenizer = self.__load_tokenizer__()

        if self.args.dataset == 'birds':
            self.args.tokenizer = self.tokenizer
            self.args.pretrained_model, self.args.preprocess = open_clip.create_model_from_pretrained(
                'local-dir:/home/models/CLIP-ViT-H-14-laion2B-s32B-b79K')
            self.train_loader, self.train_set = load_data(
                self.args, None, "train")
            self.test_loader, self.test_set = load_data(
                self.args, None, "test")
        else:
            self.train_loader, self.train_set = load_data(
                self.args, self.tokenizer, flag='train')
            self.test_loader, self.test_set = load_data(
                self.args, self.tokenizer, flag='test')
        self.accumulation_steps = max(1, int(self.args.accumulation_steps))

        self.model = PolarTaxo(self.args)

        self.optimizer = self._select_optimizer()
        self._set_device()
        self.exp_setting = "_".join([str(elem) for elem in [self.args.dataset, self.args.expID,
                                                            self.args.beta, self.args.embed_size, self.args.geometric_weight, self.args.c, self.args.vmf_margin, self.args.kappa_repel, self.args.kappa_align, self.args.kernel_setting, self.args.svgd_weight, self.args.seed]])
        self.grad_logger = GradientLogger()
        print(self.args)

    def __load_tokenizer__(self):
        """Load the tokenizer that matches the configured backbone model."""
        if self.args.dataset == 'birds':
            print("Loading CLIP Tokenizer...")
            tokenizer = open_clip.get_tokenizer(
                model_name='local-dir:/home/models/CLIP-ViT-H-14-laion2B-s32B-b79K')
        else:
            if self.args.model == 'bert':
                tokenizer = BertTokenizer.from_pretrained(
                    '/home/models/bert-base-uncased')
            elif self.args.model == 'snowflake':
                tokenizer = AutoTokenizer.from_pretrained(
                    'Snowflake/snowflake-arctic-embed-m')
        print("Tokenizer Loaded!")
        return tokenizer

    def _select_optimizer(self):
        """Select the optimizer used for Euclidean or spherical training."""
        parameters = [{"params": [p for n, p in self.model.named_parameters()],
                       "weight_decay": 0.0},]
        if self.args.implement_rectangular_opt is True:
            optimizer = optim.AdamW(
                params=parameters, lr=self.args.lr, weight_decay=0.01)
        else:

            optimizer = RiemannianAdam(
                params=parameters, lr=self.args.lr, eps=self.args.eps, betas=(0.9, 0.999))

        return optimizer

    def _set_device(self):
        """Move the model to CUDA when requested."""
        if self.args.cuda:
            self.model = self.model.cuda()

    def train_one_step(self, it, encode_parent, encode_child, encode_negative, should_step=True):

        """Run one optimization step for a mini-batch."""
        self.model.train()

        loss, taxo_loss, svgd_loss = self.model(
            it, encode_parent, encode_child, encode_negative)
        taxo_grads_to_log = None
        if it == len(self.train_loader)-1:
            final_embedding_layers = [
                ("par_projection.weight", self.model.parent_sphere.l2.weight),
                ('child_projection.weight', self.model.child_sphere.l2.weight)
            ]
            self.grad_logger.log_gradients_for_layers(
                taxo_loss, final_embedding_layers)
            taxo_grads_to_log = self.grad_logger.get_loggable_dict()

        (loss / self.accumulation_steps).backward()

        if should_step:
            self.optimizer.step()

            if self.args.implement_rectangular_opt is False:
                self.model.normalize_spherical_weights()
            self.optimizer.zero_grad()

        del encode_parent, encode_child, encode_negative

        torch.cuda.empty_cache()
        gc.collect()

        return loss, svgd_loss, taxo_grads_to_log

    def train(self, checkpoint=None, save_path=None):
        """Train PolarTaxo and periodically evaluate/save checkpoints."""
        time_tracker = []
        test_acc = test_mrr = test_wu_p = 0
        old_test_acc = old_test_mrr = old_test_wu_p = 0

        if checkpoint:
            self.model.load_state_dict(torch.load(f"{checkpoint}"))

        if save_path is None:
            savedir = os.path.join(
                "../result", self.args.dataset, "model", self.args.exp_name)
            traindir = os.path.join(
                "../result", self.args.dataset, "train", self.args.exp_name)
            if not os.path.exists(savedir):
                os.makedirs(savedir, exist_ok=True)
            if not os.path.exists(traindir):
                os.makedirs(traindir, exist_ok=True)
            save_path = os.path.join(
                "../result", self.args.dataset, "model", f"exp_model_{self.exp_setting}.checkpoint")

        for epoch in tqdm(range(self.args.epochs)):
            epoch_time = time.time()
            train_loss = []
            svgd_losses = []

            self.optimizer.zero_grad()
            for i, (encode_parent, encode_child, encode_negative) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                should_step = (
                    (i + 1) % self.accumulation_steps == 0
                    or (i + 1) == len(self.train_loader)
                )
                loss, svgd_loss, taxo_grad_log = self.train_one_step(
                    it=i, encode_parent=encode_parent, encode_child=encode_child,
                    encode_negative=encode_negative, should_step=should_step)

                train_loss.append(loss.item())
                svgd_losses.append(svgd_loss.item())

            train_loss = np.average(train_loss)
            svgd_loss = np.average(svgd_losses)
            print("Loss: ", train_loss)

            if self.args.implement_rectangular_opt is True:
                test_metrics = self.predict_rectangular()
                test_acc = test_metrics["Prec@1"]
                test_mrr = test_metrics["MRR"]
            else:
                if self.args.dataset == 'birds':
                    test_metrics = self.predict_multimodal()
                    test_acc = test_metrics['Precision']
                    test_mrr = test_metrics['mrr']
                else:
                    test_metrics = self.predict()
                    test_acc = test_metrics["Prec@1"]
                    test_mrr = test_metrics["MRR"]

            if test_acc >= old_test_acc or test_mrr >= old_test_mrr:
                final_res_dir = f"../final_result/{self.args.dataset}/{self.args.exp_name}"
                if not os.path.exists(final_res_dir):
                    os.makedirs(final_res_dir, exist_ok=True)
                final_result_save_path = f"{final_res_dir}/experiment_{self.exp_setting}.pt"
                torch.save(self.model.state_dict(), final_result_save_path)
                old_test_acc = test_acc
                old_test_mrr = test_mrr
                old_test_wu_p = test_wu_p
            time_tracker.append(time.time()-epoch_time)

            if self.args.dataset != 'birds':
                print('\nEpoch: {:04d}'.format(epoch + 1),
                      'train_loss:{:.05f}'.format(train_loss),
                      'hit@1:{:.05f}'.format(test_acc),
                      'mrr:{:.05f}'.format(test_mrr),
                      'Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                      'Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                      'Recall@10:{:0.5f}'.format(test_metrics["Recall@10"]),
                      'mr:{:.05f}'.format(test_metrics["MR"]),
                      'hit5:{:.05f}'.format(test_metrics["Prec@5"]),
                      'hit10:{:.05f}'.format(test_metrics["Prec@10"]),
                      'epoch_time:{:.01f}s'.format(time.time()-epoch_time),
                      'remain_time:{:.01f}s'.format(
                    np.mean(time_tracker)*(self.args.epochs-(1+epoch))),
                )

            if self.args.dataset == 'birds' and self.args.wandb == 1:
                wandb.log(
                    {
                        'train_loss': (train_loss),
                        'SVGD_Gradient_NORM': (svgd_loss),
                        'Precision': test_metrics['Precision'],
                        'Recall': test_metrics['Recall'],
                        'MR': test_metrics['mr'],
                        'MRR': test_metrics['mrr']
                    }
                )
            elif self.args.is_multi_parent is True and self.args.wandb == 1:
                wandb.log({
                    'train_loss': (train_loss),
                    'SVGD_Gradient_NORM': (svgd_loss),
                    'hit@1': (test_acc),
                    'mrr': (test_mrr),
                    'Recall@1': (test_metrics['Recall@1']),
                    'Recall@5': (test_metrics['Recall@5']),
                    'Recall@10': (test_metrics["Recall@10"]),
                    'BC_mr': (test_metrics["MR"]),
                    'hit@5': (test_metrics["Prec@5"]),
                    'hit@10': (test_metrics["Prec@10"]),
                })
            elif self.args.is_multi_parent is False and self.args.wandb == 1:
                wandb.log({
                    'train_loss': (train_loss),
                    'hit@1': (test_acc),
                    'mrr': (test_mrr),
                    'Recall@1': (test_metrics['Recall@1']),
                    'Recall@5': (test_metrics['Recall@5']),
                    'Recall@10': (test_metrics["Recall@10"]),
                    'mr': (test_metrics["MR"]),
                    'Wu&P': test_metrics['Wu'],
                })
            torch.save(self.model.state_dict(), os.path.join(
                traindir, f"experiment_{self.exp_setting}.checkpoint"))

            gradient_logs = f'../gradients/{self.args.dataset}/{self.args.exp_name}'
            if not os.path.exists(gradient_logs):
                os.makedirs(gradient_logs, exist_ok=True)
            with open(f'{gradient_logs}/gradient_{self.exp_setting}.json', 'a+') as f:
                json.dump(taxo_grad_log, f, indent=4)

    def get_pos_from_h_theta(self, h, theta):

        """Convert radial height and direction vectors into Cartesian positions."""
        r = self.args.r_0 * torch.exp(h)

        theta_norm = torch.norm(theta, p=2, dim=1, keepdim=True)
        theta_unit = theta / (theta_norm + 1e-8)

        pos = r * theta_unit
        return pos, r, h

    def predict_multimodal(self, tag=None, path=None
                           ):
        """Evaluate multimodal image-to-taxonomy predictions."""
        print("Prediction starting....")
        store_csv = False
        if tag == 'test' and path:
            self.model.load_state_dict(torch.load(path))
            store_csv = True

        self.model.eval()
        with torch.no_grad():
            score_list = []
            gt_label = self.test_set.test_gt_id

            q_sphere = self.model.multimodal_child_projection(
                self.test_set.encode_query)
            q_k = self.model.vmf_regulariser.kappa_predictor(q_sphere)
            q_mu = self.model.vmf_regulariser.mu_predictor(q_sphere)

            candidates_sphere = list()
            candidates_k = list()
            candidates_mu = list()

            for encode_candidate in self.test_loader:
                candidate_sphere = self.model.multimodal_parent_projection(
                    encode_candidate)
                candidate_k = self.model.vmf_regulariser.kappa_predictor(
                    candidate_sphere)
                candidate_mu = self.model.vmf_regulariser.mu_predictor(
                    candidate_sphere)

                candidates_sphere.append(candidate_sphere)
                candidates_k.append(candidate_k)
                candidates_mu.append(candidate_mu)

            candidates_sphere = torch.cat(candidates_sphere, dim=0)
            candidates_k = torch.cat(candidates_k, dim=0)
            candidates_mu = torch.cat(candidates_mu, dim=0)

            num_queries = q_sphere.size(0)
            num_candidates = candidates_sphere.size(0)

            for i in tqdm(range(num_queries), desc='Evaluating Queries'):
                q_sph = q_sphere[i].unsqueeze(0).expand(num_candidates, -1)

                geometric_score = torch.sum(candidates_sphere*q_sph, dim=1)

                final_score = geometric_score

                score_list.append(final_score)

            score_matrix = torch.stack(score_list, dim=0)
            print("Score matrix size:", score_matrix.size())
            sorted_scores, indices = score_matrix.sort(dim=1, descending=True)
            print(sorted_scores[:, :5])

            candidate_list = np.array(list(self.test_set.true_concept_set))
            test_metrics = metrics_multi_modal(
                indices, gt_label, candidate_list, self.test_set.id_concept, self.test_set.test_concepts_id)

            print('Precision:{:.05f}'.format(test_metrics["Precision"]),
                  'Recall:{:.05f}'.format(test_metrics["Recall"]),
                  "F1:{:.05f}".format(test_metrics['f1']),
                  "MRR:{:.05f}".format(test_metrics['mrr']),
                  "mr:{:.05f}".format(test_metrics['mr']))

        return test_metrics

    def predict_rectangular(self, tag=None, path=None):
        """Evaluate the rectangular polar-coordinate ablation."""
        print("prediction starting....")

        store_csv = False
        if tag == "test" and path:
            self.model.load_state_dict(torch.load(path))
            store_csv = True

        self.model.eval()
        with torch.no_grad():
            score_list = []
            gt_label = self.test_set.test_gt_id

            q_psi, q_theta = self.model.direct_projection_child(
                self.test_set.encode_query)
            q_angles = torch.cat([q_psi, q_theta], dim=1)
            q_sphere = spherical_to_cartesian(q_angles)

            candidates_sphere = list()

            for encode_candidate in self.test_loader:
                candidate_psi, candidate_theta = self.model.direct_projection_parent(
                    encode_candidate)
                candidate_angles = torch.cat(
                    [candidate_psi, candidate_theta], dim=1)
                candidate_sphere = spherical_to_cartesian(candidate_angles)

                candidates_sphere.append(candidate_sphere)
            candidates_sphere = torch.cat(candidates_sphere, dim=0)

            num_queries = q_psi.size(0)
            num_candidates = candidates_sphere.size(0)

            for i in tqdm(range(num_queries), desc='evaluating queries'):
                q_sphere = q_sphere[i].unsqueeze(0).expand(num_candidates, -1)

                score = torch.sum(q_sphere*candidates_sphere, dim=1)
                score_list.append(score)

            score_matrix = torch.stack(score_list, dim=0)
            print("Score matrix size:", score_matrix.size())
            sorted_scores, indices = score_matrix.sort(dim=1, descending=True)
            print(sorted_scores[:, :5])

            if self.args.is_multi_parent is True:
                candidate_list = np.array(list(self.test_set.true_concept_set))
                test_metrics = metrics_multi_p(
                    indices, gt_label, candidate_list, self.test_set.id_concept, self.test_set.test_concepts_id)

                print('Hit@1:{:.05f}'.format(test_metrics["Prec@1"]),
                      'mrr:{:.05f}'.format(test_metrics["MRR"]),
                      'Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                      'mr:{:.05f}'.format(test_metrics["MR"]),
                      'Hit@5:{:.05f}'.format(test_metrics["Prec@5"]),
                      'Hit@10:{:.05f}'.format(test_metrics["Prec@10"]),
                      'Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                      'Recall@10: {:.05f}'.format(test_metrics["Recall@10"]))
            else:
                test_metrics = metrics(
                    indices,
                    gt_label,
                    self.train_set.train_concept_set,
                    self.test_set.path2root,
                    self.test_set.id_concept,
                    self.train_set.id_concept,
                    self.test_set.test_concepts_id,
                    sorted_scores
                )

                print('Hit@1:{:.05f}'.format(test_metrics["Prec@1"]),
                      'mrr:{:.05f}'.format(test_metrics["MRR"]),
                      'Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                      'mr:{:.05f}'.format(test_metrics["MR"]),
                      'prec@5:{:.05f}'.format(test_metrics["Prec@5"]),
                      'prec@10:{:.05f}'.format(test_metrics["Prec@10"]),
                      'Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                      'Recall@10: {:.05f}'.format(test_metrics["Recall@10"]))

        results_json_dir = f'../results/{self.args.dataset}/{self.args.exp_name}'
        if not os.path.exists(results_json_dir):
            os.makedirs(results_json_dir, exist_ok=True)
        with open(f'../results/{self.args.dataset}/{self.args.exp_name}/res_{self.exp_setting}.json', 'a+') as f:
            d = vars(self.args)
            expt_details = {
                "Arguments": d,
                "Test Metrics": test_metrics,
            }
            json.dump(expt_details, f, indent=4)

        return test_metrics

    def predict(self, tag=None, path=None):
        """Evaluate taxonomy prediction by ranking candidate spherical similarities."""
        print("Prediction starting.....")
        store_csv = False
        if tag == "test" and path:
            self.model.load_state_dict(torch.load(path))
            store_csv = True

        self.model.eval()
        with torch.no_grad():
            score_list = []
            gt_label = self.test_set.test_gt_id

            q_sphere = self.model.child_projection(
                self.test_set.encode_query)
            q_k = self.model.vmf_regulariser.kappa_predictor(
                q_sphere)
            q_mu = self.model.vmf_regulariser.mu_predictor(
                q_sphere)

            candidates_sphere = list()
            candidates_k = list()
            candidates_mu = list()
            for encode_candidate in self.test_loader:
                candidate_sphere = self.model.par_projection(
                    encode_candidate)
                candidate_k = self.model.vmf_regulariser.kappa_predictor(
                    candidate_sphere)
                candidate_mu = self.model.vmf_regulariser.mu_predictor(
                    candidate_sphere)

                candidates_sphere.append(candidate_sphere)
                candidates_k.append(candidate_k)
                candidates_mu.append(candidate_mu)

            candidates_sphere = torch.cat(candidates_sphere, dim=0)
            candidates_k = torch.cat(candidates_k, dim=0)
            candidates_mu = torch.cat(candidates_mu, dim=0)

            num_queries = q_sphere.size(0)
            num_candidates = candidates_sphere.size(0)

            for i in tqdm(range(num_queries), desc='Evaluating Queries'):

                q_sph = q_sphere[i].unsqueeze(0).expand(num_candidates, -1)
                q_mu = q_mu[i].unsqueeze(0).expand(num_candidates, -1)
                q_k = q_k[i].unsqueeze(0).expand(num_candidates, -1)

                geometric_score = torch.sum(candidates_sphere*q_sph, dim=1)

                final_score = geometric_score

                score_list.append(final_score)

            score_matrix = torch.stack(score_list, dim=0)
            print("Score matrix size:", score_matrix.size())
            sorted_scores, indices = score_matrix.sort(dim=1, descending=True)
            print(sorted_scores[:, :5])

            if self.args.is_multi_parent is True:
                candidate_list = np.array(list(self.test_set.true_concept_set))
                test_metrics = metrics_multi_p(
                    indices, gt_label, candidate_list, self.test_set.id_concept, self.test_set.test_concepts_id)

                print('Hit@1:{:.05f}'.format(test_metrics["Prec@1"]),
                      'mrr:{:.05f}'.format(test_metrics["MRR"]),
                      'Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                      'mr:{:.05f}'.format(test_metrics["MR"]),
                      'Hit@5:{:.05f}'.format(test_metrics["Prec@5"]),
                      'Hit@10:{:.05f}'.format(test_metrics["Prec@10"]),
                      'Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                      'Recall@10: {:.05f}'.format(test_metrics["Recall@10"]))
            else:
                test_metrics = metrics(
                    indices,
                    gt_label,
                    self.train_set.train_concept_set,
                    self.test_set.path2root,
                    self.test_set.id_concept,
                    self.train_set.id_concept,
                    self.test_set.test_concepts_id,
                    sorted_scores
                )

                print('Hit@1:{:.05f}'.format(test_metrics["Prec@1"]),
                      'mrr:{:.05f}'.format(test_metrics["MRR"]),
                      'Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                      'mr:{:.05f}'.format(test_metrics["MR"]),
                      'prec@5:{:.05f}'.format(test_metrics["Prec@5"]),
                      'prec@10:{:.05f}'.format(test_metrics["Prec@10"]),
                      'Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                      'Recall@10: {:.05f}'.format(test_metrics["Recall@10"]))

        results_json_dir = f'../results/{self.args.dataset}/{self.args.exp_name}'
        if not os.path.exists(results_json_dir):
            os.makedirs(results_json_dir, exist_ok=True)
        with open(f'../results/{self.args.dataset}/{self.args.exp_name}/res_{self.exp_setting}.json', 'a+') as f:
            d = vars(self.args)
            expt_details = {
                "Arguments": d,
                "Test Metrics": test_metrics,
            }
            json.dump(expt_details, f, indent=4)

        return test_metrics

    def level_wise_prediction(self, tag=None, path=None):
        """Evaluate level-aware orbital predictions using angular and radial scores."""
        print("Prediction starting....")
        store_csv = False

        if tag == 'test' and path:
            self.model.load_state_dict(torch.load(path))
            store_csv = True

        self.model.eval()
        with torch.no_grad():
            score_list = list()
            query_radii_list = list()

            gt_label = self.test_set.test_gt_id
            q_z = self.model.child_projection(self.test_set.encode_query)
            q_mu_all = self.model.vmf_regulariser.mu_predictor(q_z)
            q_k_all = self.model.vmf_regulariser.kappa_predictor(q_z)

            candidates_sphere = list()
            candidates_mu_list = list()
            candidates_k_list = list()

            if self.args.is_multi_parent == True:

                candidate_list = np.array(
                    sorted(list(self.test_set.true_concept_set)))
            else:
                candidate_list = np.array(
                    sorted(list(self.train_set.train_concept_set)))

            for encode_candidate in self.test_loader:
                candidate_z = self.model.par_projection(encode_candidate)
                candidate_mu = self.model.vmf_regulariser.mu_predictor(
                    candidate_z)
                candidate_k = self.model.vmf_regulariser.kappa_predictor(
                    candidate_z)

                candidates_sphere.append(candidate_z)
                candidates_k_list.append(candidate_k)
                candidates_mu_list.append(candidate_mu)

            candidates_mu = torch.cat(candidates_mu_list, dim=0)
            candidates_sphere = torch.cat(candidates_sphere, dim=0)
            candidates_k = torch.cat(candidates_k_list, dim=0)

            num_queries = q_z.size(0)
            num_candidates = candidates_k.size(0)

            candidate_radii_tensor = torch.tensor([
                self.test_set.levels[cid]['radii'] for cid in candidate_list
            ]).to(device=q_z.device)
            candidate_depths = torch.tensor([self.test_set.levels[cid]['depth']
                                             for cid in candidate_list]).to(q_z.device)
            candidate_descendants = torch.tensor([
                self.test_set.levels[cid]['descendents'] for cid in candidate_list]).to(q_z.device)

            if self.test_set.levels:
                first_level_item = next(iter(self.test_set.levels.values()))
                score_min = first_level_item['raw_score_min']
                score_range = first_level_item['raw_score_range']
            else:
                score_min = 0
                score_range = 1
            candidate_radii_list = list()
            start = time.time()
            for i in tqdm(range(num_queries), desc='evaluating queries'):
                q_sph = q_z[i].unsqueeze(0).expand(num_candidates, -1)

                dot_product = torch.sum(candidates_sphere * q_sph, dim=1)

                norm_candidates = torch.norm(candidates_sphere, p=2, dim=1)
                norm_q = torch.norm(q_z[i], p=2)

                epsilon = 1e-8

                angular_score = dot_product / \
                    ((norm_q * norm_candidates) + epsilon)

                query_radius = (
                    candidate_depths+1)
                candidate_updated_radius = (
                    candidate_depths)+((torch.log1p(candidate_descendants+1))/torch.log(torch.tensor(2)))
                query_radius_normalized = 1 - \
                    ((query_radius-score_min)/(score_range))
                candidate_radius_normalized = 1 - \
                    ((candidate_updated_radius-score_min)/(score_range))

                radius_diff = torch.abs(
                    candidate_radius_normalized-query_radius_normalized)

                radius_score = torch.where(
                    angular_score > 1-(self.args.potential_strength*(radius_diff**2)), 1.0, 0.0)
                final_score = radius_score*angular_score

                score_list.append(final_score)
                candidate_radii_list.append(candidate_radius_normalized)
                query_radii_list.append(query_radius_normalized)
            end = time.time()
            print(candidate_radii_list[2][:10])

            score_matrix = torch.stack(score_list, dim=0)
            print("Score matrix size:", score_matrix.size())
            sorted_scores, indices = score_matrix.sort(dim=1, descending=True)
            print(sorted_scores[:, :5])

            print("Generating static solar system plots for top 5 predictions...")
            num_queries_to_plot = 5
            num_preds_to_plot = 5

            for i in range(min(num_queries_to_plot, num_queries)):
                query_id = self.test_set.test_concepts_id[i]

                gt_ids = self.test_set.test_gt_id[i]
                ground_truth_info = []
                if self.args.is_multi_parent is False:
                    gt_ids = [gt_ids]
                if gt_ids:
                    gt_indices = [np.where(candidate_list == gid)[0][0]
                                  for gid in gt_ids if gid in candidate_list]
                    for gt_idx in gt_indices:
                        gt_id = candidate_list[gt_idx]
                        ground_truth_info.append({
                            'name': self.test_set.id_concept[gt_id],
                            'radius': candidate_radii_tensor[gt_idx].item(),
                            'embedding': candidates_sphere[gt_idx].cpu().numpy()
                        })

                top_indices = indices[i, :num_preds_to_plot].cpu().numpy()
                for rank, pred_idx in enumerate(top_indices):
                    pred_id = candidate_list[pred_idx]

                    query_info = {
                        'name': self.test_set.id_concept[query_id],
                        'radius': query_radii_list[i][pred_idx].item(),
                        'embedding': q_z[i].cpu().numpy()
                    }

                    predicted_info = {
                        'name': self.test_set.id_concept[pred_id],
                        'radius': candidate_radii_list[i][pred_idx].item(),
                        'embedding': candidates_sphere[pred_idx].cpu().numpy()
                    }

                    plot_save_path = f'../results/{self.args.dataset}/plots/query_{query_id}_vs_pred_rank_{rank+1}.png'
                    plot_concept_space_map(
                        query_info, predicted_info, ground_truth_info, plot_save_path)

                    print(
                        f"Plots saved in '../results/{self.args.dataset}/plots/'")

            if self.args.is_multi_parent is True:
                candidate_list = np.array(list(self.test_set.true_concept_set))
                test_metrics = metrics_multi_p(
                    indices, gt_label, candidate_list, self.test_set.id_concept, self.test_set.test_concepts_id)

                print('Hit@1:{:.05f}'.format(test_metrics["Prec@1"]),
                      'mrr:{:.05f}'.format(test_metrics["MRR"]),
                      'Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                      'mr:{:.05f}'.format(test_metrics["MR"]),
                      'Hit@5:{:.05f}'.format(test_metrics["Prec@5"]),
                      'Hit@10:{:.05f}'.format(test_metrics["Prec@10"]),
                      'Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                      'Recall@10: {:.05f}'.format(test_metrics["Recall@10"]))

                print("Time Interval: ", end-start)
            else:
                test_metrics = metrics(
                    indices,
                    gt_label,
                    self.train_set.train_concept_set,
                    self.test_set.path2root,
                    self.test_set.id_concept,
                    self.train_set.id_concept,
                    self.test_set.test_concepts_id,
                    sorted_scores
                )

                print('Hit@1:{:.05f}'.format(test_metrics["Prec@1"]),
                      'mrr:{:.05f}'.format(test_metrics["MRR"]),
                      'Recall@1:{:.05f}'.format(test_metrics["Recall@1"]),
                      'mr:{:.05f}'.format(test_metrics["MR"]),
                      'prec@5:{:.05f}'.format(test_metrics["Prec@5"]),
                      'prec@10:{:.05f}'.format(test_metrics["Prec@10"]),
                      'Recall@5:{:.05f}'.format(test_metrics["Recall@5"]),
                      'Recall@10: {:.05f}'.format(test_metrics["Recall@10"]))

        with open(f'../results/{self.args.dataset}/res_{self.exp_setting}.json', 'a+') as f:
            d = vars(self.args)
            expt_details = {
                "Arguments": d,
                "Test Metrics": test_metrics,
            }
            json.dump(expt_details, f, indent=4)

        return test_metrics

    def visualize_angle_distributions(self, tag=None, path=None):

        """Plot distributions of learned candidate spherical angles."""
        print("Starting global distribution analysis for Candidates...")
        if tag == "test" and path:
            print(f"Loading model from: {path}")
            self.model.load_state_dict(torch.load(path))

        sns.set_context("poster", font_scale=1.2)
        sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
        plt.rcParams.update({
            'font.family': 'serif', 'font.weight': 'bold',
            'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
            'axes.titlesize': 28, 'axes.labelsize': 24,
            'xtick.labelsize': 20, 'ytick.labelsize': 20,
            'legend.fontsize': 22, 'mathtext.fontset': 'dejavuserif',
        })

        candidate_color = '#029E73'

        self.model.eval()
        with torch.no_grad():
            candidates_sphere_list = []
            print("Processing all candidates...")
            for encode_candidate in tqdm(self.test_loader, desc='Loading Candidates'):
                candidate_z = self.model.par_projection(encode_candidate)
                candidate_sphere = self.model.vmf_regulariser.mu_predictor(
                    candidate_z)
                candidates_sphere_list.append(candidate_sphere)

            candidates_sphere = torch.cat(candidates_sphere_list, dim=0)

            print("All embeddings loaded. Calculating angles...")

            c_thetas, c_psi1s, c_psi2s, c_psid = cartesian_to_spherical_angles(
                candidates_sphere)

            all_data = {
                'candidate_theta': c_thetas.cpu().numpy(), 'candidate_psi1': c_psi1s.cpu().numpy(),
                'candidate_psi2': c_psi2s.cpu().numpy(), 'candidate_psid': c_psid.cpu().numpy(),
            }

            print("Generating combined distribution plots...")
            fig, axes = plt.subplots(1, 4, figsize=(26, 7), dpi=300)

            plot_config = {
                'kde': True, 'bins': 50, 'stat': 'density',
                'element': 'step', 'fill': True, 'alpha': 0.5, 'linewidth': 2.0,
            }

            plot_titles = [r"Longitudinal ($\theta$)", r"1st Latitudinal ($\psi_1$)",
                           r"2nd Latitudinal ($\psi_2$)", r"Last Latitudinal ($\psi_{d-1}$)"]
            plot_data_keys = ['candidate_theta', 'candidate_psi1',
                              'candidate_psi2', 'candidate_psid']
            x_limits = [(0, 2 * np.pi), (0, np.pi), (0, np.pi), (0, np.pi)]

            for i, ax in enumerate(axes):
                sns.histplot(all_data[plot_data_keys[i]],
                             ax=ax, **plot_config, color=candidate_color)

                ax.set_title(plot_titles[i])
                ax.set_xlim(x_limits[i])
                ax.set_ylabel('Density')
                ax.set_xlabel('Angle (radians)')

                if i == 0:
                    ax.axhline(y=1/(2*np.pi), color='gray',
                               linestyle='--', alpha=0.7)
                else:
                    ax.axvline(x=np.pi/2, color='red',
                               linestyle=':', alpha=0.7, linewidth=2.5)

            candidate_patch = mpatches.Patch(
                color=candidate_color, alpha=0.5, label='Candidates')
            equator_line = mlines.Line2D(
                [], [], color='red', linestyle=':', label='Equator')

            fig.legend(handles=[candidate_patch, equator_line],
                       loc='upper center', bbox_to_anchor=(0.5, 1.1),
                       ncol=2, frameon=False)

            plt.tight_layout()
            save_path = f"candidate_angle_distributions_{self.args.dataset}.pdf"
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
            plt.close()

    def generate_embeddings_and_plot(self, path=None, n_clusters=15):
        """Generate candidate embeddings and save a UMAP cluster plot."""
        print("Visualization: Loading model and generating embeddings...")

        if path:
            print(f"Loading weights from {path}")
            self.model.load_state_dict(torch.load(path))

        self.model.eval()

        with torch.no_grad():
            candidates_sphere = list()

            if self.args.dataset in ['mesh', 'wordnet_verb', 'semeval_food']:
                concept_ids = np.array(
                    sorted(list(self.test_set.true_concept_set)))
            else:
                concept_ids = np.array(
                    sorted(list(self.train_set.train_concept_set)))

            for encode_candidate in self.test_loader:
                candidate_z = self.model.par_projection(encode_candidate)
                candidates_sphere.append(candidate_z)

            candidates_sphere = torch.cat(candidates_sphere, dim=0)

        save_path = f'{self.args.dataset}_concept_clusters_umap.png'

        visualize_concept_clusters(
            embeddings=candidates_sphere,
            concept_ids=concept_ids,
            id_to_name_map=self.test_set.id_concept,
            save_path=save_path,
            n_clusters=n_clusters
        )

    def analyze_variances(self, tag=None, path=None):

        """Analyze and plot projection-layer weight variance."""
        print("Starting variance analysis...")
        if tag == "test" and path:
            print(f"Loading model from: {path}")
            self.model.load_state_dict(torch.load(path))

        layers_to_analyze = {
            "Projection Layer 1": self.model.parent_sphere.l1.weight.data.cpu().numpy(),
            "Projection Later 2": self.model.parent_sphere.l2.weight.data.cpu().numpy()
        }

        variances = {}
        all_weights = []
        labels = []

        print("\n--- Weight Variance Analysis ---")
        for name, weights in layers_to_analyze.items():
            flat_weights = weights.flatten()

            var = np.var(flat_weights)
            mean = np.mean(flat_weights)
            variances[name] = var

            all_weights.append(flat_weights)
            labels.append(name)

            print(f"{name}: Mean = {mean:.5f}, Variance = {var:.5f}")

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        sns.boxplot(data=all_weights, palette="Set2")
        plt.xticks(ticks=range(len(labels)), labels=labels, fontsize=12)
        plt.ylabel("Weight Value", fontsize=12)
        plt.title(
            f"Distribution of Projection Layer Weights ({self.args.dataset})", fontsize=14)

        for i, name in enumerate(labels):
            plt.text(i, np.max(all_weights[i]), f"Var: {variances[name]:.4f}",
                     ha='center', va='bottom', fontweight='bold', color='black')

        save_path = f"weight_variance_analysis_{self.args.dataset}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Variance plot saved to {save_path}")

    def generate_case_study(self, tag='test', path=None, output_path='case_studies.csv'):

        """Write a CSV with top-ranked predictions for qualitative case studies."""
        print(f"Generating Case Study CSV at: {output_path}...")

        output_path = f"{self.args.dataset}_case_studies.csv"
        if tag == 'test' and path:
            print(f"Loading weights from {path}")
            self.model.load_state_dict(torch.load(path))

        self.model.eval()

        with torch.no_grad():
            q_z = self.model.child_projection(self.test_set.encode_query)

            candidates_sphere_list = []

            if self.args.is_multi_parent:
                candidate_list = np.array(
                    sorted(list(self.test_set.true_concept_set)))
            else:
                candidate_list = np.array(
                    sorted(list(self.train_set.train_concept_set)))

            for encode_candidate in tqdm(self.test_loader, desc="Encoding Candidates"):
                candidate_z = self.model.par_projection(encode_candidate)
                candidates_sphere_list.append(candidate_z)

            candidates_sphere = torch.cat(candidates_sphere_list, dim=0)

            candidate_depths = torch.tensor([
                self.test_set.levels[cid]['depth'] for cid in candidate_list
            ]).to(q_z.device)

            candidate_descendants = torch.tensor([
                self.test_set.levels[cid]['descendents'] for cid in candidate_list
            ]).to(q_z.device)

            score_min = self.test_set.levels[0]['raw_score_min']
            score_range = self.test_set.levels[0]['raw_score_range']

            num_queries = q_z.size(0)
            num_candidates = candidates_sphere.size(0)

            with open(output_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                header = ['Query ID', 'Query Name', 'Ground Truths',
                          'Pred 1', 'Pred 2', 'Pred 3', 'Pred 4', 'Pred 5']
                writer.writerow(header)

                for i in tqdm(range(num_queries), desc="Processing Case Studies"):
                    q_sph = q_z[i].unsqueeze(0).expand(num_candidates, -1)

                    dot_product = torch.sum(candidates_sphere * q_sph, dim=1)
                    norm_candidates = torch.norm(candidates_sphere, p=2, dim=1)
                    norm_q = torch.norm(q_z[i], p=2)
                    angular_score = dot_product / \
                        ((norm_q * norm_candidates) + 1e-8)

                    query_radius = candidate_depths + 1
                    candidate_updated_radius = (candidate_depths + 1) + \
                        ((torch.log1p(candidate_descendants)) /
                         torch.log(torch.tensor(2.0)))

                    q_rad_norm = 1 - ((query_radius - score_min) / score_range)
                    c_rad_norm = 1 - \
                        ((candidate_updated_radius - score_min) / score_range)

                    radius_diff = torch.abs(c_rad_norm - q_rad_norm)

                    radius_score = torch.where(
                        angular_score > 1-(self.args.potential_strength*(radius_diff**2)), 1.0, 0.0)
                    final_score = angular_score - \
                        self.args.potential_strength*(radius_diff**2)

                    top_vals, top_indices = torch.topk(final_score, k=5)
                    top_indices = top_indices.cpu().numpy()

                    query_id = self.test_set.test_concepts_id[i]
                    query_name = self.test_set.id_concept[query_id]

                    gt_ids = self.test_set.test_gt_id[i]
                    if not isinstance(gt_ids, list) and not isinstance(gt_ids, np.ndarray):
                        gt_ids = [gt_ids]

                    gt_names = [self.test_set.id_concept.get(
                        gid, str(gid)) for gid in gt_ids if gid in self.test_set.id_concept]
                    gt_str = "; ".join(gt_names)

                    pred_names = []
                    for idx in top_indices:
                        pred_id = candidate_list[idx]
                        p_name = self.test_set.id_concept.get(
                            pred_id, str(pred_id))
                        pred_names.append(p_name)

                    row = [query_id, query_name, gt_str] + pred_names
                    writer.writerow(row)

        print(f"Case studies saved successfully to {output_path}")

    def plot_kappa_vs_depth(self, tag=None, path=None):
        """Plot predicted vMF concentration against taxonomy depth."""
        print("Starting Kappa vs. Depth Analysis (Aggregated)...")

        if path:
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        candidates_k_list = list()

        if self.args.is_multi_parent == True:
            candidate_list = np.array(
                sorted(list(self.test_set.true_concept_set)))
        else:
            candidate_list = np.array(
                sorted(list(self.train_set.train_concept_set)))

        with torch.no_grad():
            for encode_candidate in tqdm(self.test_loader, desc="Extracting Kappas"):
                candidate_z = self.model.par_projection(encode_candidate)
                candidate_k = self.model.vmf_regulariser.kappa_predictor(
                    candidate_z)
                candidates_k_list.append(candidate_k)

        candidates_k = torch.cat(
            candidates_k_list, dim=0).squeeze().cpu().numpy()

        min_len = min(len(candidates_k), len(candidate_list))
        candidates_k = candidates_k[:min_len]
        candidate_list = candidate_list[:min_len]

        candidate_depths = []
        for cid in candidate_list:
            candidate_depths.append(self.test_set.levels[cid]['depth'])
        candidate_depths = np.array(candidate_depths)

        df = pd.DataFrame({
            'Depth': candidate_depths,
            'Kappa': candidates_k
        })

        median_stats = df.groupby('Depth')['Kappa'].median().reset_index()

        raw_corr, raw_p = pearsonr(df['Depth'], df['Kappa'])

        median_corr, median_p = pearsonr(
            median_stats['Depth'], median_stats['Kappa'])

        print(f"\n=== Statistical Results ===")
        print(
            f"1. Global Raw Correlation (N={len(df)}): {raw_corr:.4f} (p={raw_p:.2e})")
        print(
            f"2. Aggregated Median Trend Correlation (N={len(median_stats)} levels): {median_corr:.4f} (p={median_p:.4f})")

        save_dir = f'./depth'
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))

        ax = sns.boxplot(x='Depth', y='Kappa', data=df,
                         color="#89CFF0",
                         showfliers=False,
                         width=0.6,
                         linewidth=1.2)

        sns.pointplot(x='Depth', y='Kappa', data=median_stats,
                      color='darkblue', markers='o', scale=0.6, linestyles='-', ax=ax)

        plt.title(
            f'Semantic Certainty vs. Depth: {self.args.dataset}\n(Median Trend Correlation: {median_corr:.2f})')
        plt.xlabel('Hierarchical Depth')
        plt.ylabel(r'Concentration Parameter ($\kappa$)')
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        filename = f"kappa_vs_depth_{self.args.dataset}.pdf"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Clean plot saved to: {save_path}")
