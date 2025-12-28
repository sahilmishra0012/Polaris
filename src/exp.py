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
from optimizer import RiemannianAdam
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import gc
import seaborn as sns
from hooks import GradientLogger
import csv
import wandb

os.environ["WANDB_MODE"] = "online"


class Experiments(object):

    def __init__(self, args):
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
        self.accumulation_steps = self.args.accumulation_steps

        self.model = PolarTaxo(self.args)

        self.optimizer = self._select_optimizer()
        self._set_device()
        self.exp_setting = "_".join([str(elem) for elem in [self.args.dataset, self.args.expID,
                                                            self.args.beta, self.args.embed_size, self.args.geometric_weight, self.args.c, self.args.vmf_margin, self.args.kappa_repel, self.args.kappa_align, self.args.kernel_setting, self.args.svgd_weight, self.args.seed]])
        self.grad_logger = GradientLogger()
        # setting = {
        #     "dataset": self.args.dataset,
        #     "expID": self.args.expID,
        #     'beta': self.args.beta,
        #     'embed_size': self.args.embed_size,
        #     'seed': self.args.seed,
        #     'c': self.args.c
        # }
        print(self.args)

    def __load_tokenizer__(self):
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
        parameters = [{"params": [p for n, p in self.model.named_parameters()],
                       "weight_decay": 0.0},]

        if self.args.optim == "adam":
            optimizer = optim.Adam(parameters, lr=self.args.lr)
        elif self.args.optim == "adamw":
            optimizer = RiemannianAdam(
                params=parameters, lr=self.args.lr, eps=self.args.eps, betas=(0.9, 0.999))

        return optimizer

    def _set_device(self):
        if self.args.cuda:
            self.model = self.model.cuda()

    def train_one_step(self, it, encode_parent, encode_child, encode_negative):

        self.model.train()

        loss, taxo_loss, svgd_loss = self.model(
            it, encode_parent, encode_child, encode_negative)
        taxo_grads_to_log = None
        if it == len(self.train_loader)-1:
            # Calculate gradient norm for taxonomy and geometric loss
            final_embedding_layers = [
                ("par_projection.weight", self.model.parent_sphere.l2.weight),
                ('child_projection.weight', self.model.child_sphere.l2.weight)
            ]
            self.grad_logger.log_gradients_for_layers(
                taxo_loss, final_embedding_layers)
            taxo_grads_to_log = self.grad_logger.get_loggable_dict()

        loss.backward()

        self.optimizer.step()

        self.model.normalize_spherical_weights()
        self.optimizer.zero_grad()

        del encode_parent, encode_child, encode_negative

        torch.cuda.empty_cache()
        gc.collect()

        return loss, svgd_loss, taxo_grads_to_log

    def train(self, checkpoint=None, save_path=None):
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
                loss, svgd_loss, taxo_grad_log = self.train_one_step(
                    it=i, encode_parent=encode_parent, encode_child=encode_child, encode_negative=encode_negative)

                train_loss.append(loss.item())
                svgd_losses.append(svgd_loss.item())

            train_loss = np.average(train_loss)
            svgd_loss = np.average(svgd_losses)
            print("Loss: ", train_loss)

            if self.args.dataset == 'birds':
                test_metrics = self.predict_multimodal()
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

            if self.args.is_multi_parent is True and self.args.wandb == 1:
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

            # Log to wandb and keep a copy locally
            # if taxo_grad_log is not None:
            #     wandb.log(taxo_grad_log)

            gradient_logs = f'../gradients/{self.args.dataset}/{self.args.exp_name}'
            if not os.path.exists(gradient_logs):
                os.makedirs(gradient_logs, exist_ok=True)
            with open(f'{gradient_logs}/gradient_{self.exp_setting}.json', 'a+') as f:
                json.dump(taxo_grad_log, f, indent=4)

    def get_pos_from_h_theta(self, h, theta):

        r = self.args.r_0 * torch.exp(h)

        theta_norm = torch.norm(theta, p=2, dim=1, keepdim=True)
        theta_unit = theta / (theta_norm + 1e-8)

        pos = r * theta_unit
        return pos, r, h

    def predict_multimodal(self, tag=None, path=None
                           ):
        print("Prediction starting....")
        store_csv = False
        if tag == 'test' and path:
            self.model.load_state_dict(torch.load(path))
            store_csv = True

        self.model.eval()
        with torch.no_grad():
            score_list = []
            gt_label = self.test_set.test_gt_id

            # Query is an image embedding batch
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
                candidate_mu = self.model.vmf_regulariser.kappa_predictor(
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
                # q_mu = q_mu[i].unsqueeze(0).expand(num_candidates, -1)
                # q_k = q_k[i].unsqueeze(0).expand(num_candidates, -1)

                geometric_score = torch.sum(candidates_sphere*q_sph, dim=1)

                final_score = geometric_score

                score_list.append(final_score)

            score_matrix = torch.stack(score_list, dim=0)
            print("Score matrix size:", score_matrix.size())
            sorted_scores, indices = score_matrix.sort(dim=1, descending=True)
            print(sorted_scores[:, :5])

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

        # results_json_dir = f'../results/{self.args.dataset}/{self.args.exp_name}'
        # if not os.path.exists(results_json_dir):
        #     os.makedirs(results_json_dir, exist_ok=True)
        # with open(f'../results/{self.args.dataset}/{self.args.exp_name}/res_{self.exp_setting}.json', 'a+') as f:
        #     d = vars(self.args)
        #     expt_details = {
        #         "Arguments": d,
        #         "Test Metrics": test_metrics,
        #     }
        #     json.dump(expt_details, f, indent=4)

        return test_metrics

    def predict(self, tag=None, path=None):
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

            candidate_list = np.array(
                sorted(list(self.test_set.true_concept_set)))
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
            score_min = self.test_set.levels[0]['raw_score_min']
            score_range = self.test_set.levels[0]['raw_score_range']

            candidate_radii_list = list()
            for i in tqdm(range(num_queries), desc='evaluating queries'):
                q_sph = q_z[i].unsqueeze(0).expand(num_candidates, -1)

                dot_product = torch.sum(candidates_sphere * q_sph, dim=1)

                norm_candidates = torch.norm(candidates_sphere, p=2, dim=1)
                norm_q = torch.norm(q_z[i], p=2)

                epsilon = 1e-8

                angular_score = dot_product / \
                    ((norm_q * norm_candidates) + epsilon)

                query_radius = candidate_depths+1
                candidate_updated_radius = (
                    candidate_depths+1)+((torch.log1p(candidate_descendants))/torch.log(torch.tensor(2)))
                query_radius_normalized = 1 -\
                    ((query_radius-score_min)/(score_range))
                candidate_radius_normalized = 1 -\
                    ((candidate_updated_radius-score_min)/(score_range))

                radius_diff = torch.abs(
                    candidate_radius_normalized-query_radius_normalized)

                radius_score = torch.where(
                    angular_score > 1-(radius_diff**2), 1.0, 0.0)
                final_score = radius_score*angular_score

                score_list.append(final_score)
                candidate_radii_list.append(candidate_radius_normalized)
                query_radii_list.append(query_radius_normalized)

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

            print(f"Plots saved in '../results/{self.args.dataset}/plots/'")

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

        with open(f'../results/{self.args.dataset}/res_{self.exp_setting}.json', 'a+') as f:
            d = vars(self.args)
            expt_details = {
                "Arguments": d,
                "Test Metrics": test_metrics,
            }
            json.dump(expt_details, f, indent=4)

        return test_metrics

    def visualize_angle_distributions(self, tag=None, path=None):

        print("Starting global distribution analysis...")
        if tag == "test" and path:
            print(f"Loading model from: {path}")
            self.model.load_state_dict(torch.load(path))

        self.model.eval()
        with torch.no_grad():
            q_z = self.model.child_projection(self.test_set.encode_query)
            q_sphere = self.model.vmf_regulariser.mu_predictor(q_z)
            q_k = self.model.vmf_regulariser.kappa_predictor(q_z)

            candidates_sphere_list = []
            candidates_k_list = []
            print("Processing all candidates...")
            for encode_candidate in tqdm(self.test_loader, desc='Loading Candidates'):
                candidate_z = self.model.par_projection(encode_candidate)
                candidate_sphere = self.model.vmf_regulariser.mu_predictor(
                    candidate_z)
                candidate_k = self.model.vmf_regulariser.kappa_predictor(
                    candidate_z)
                candidates_sphere_list.append(candidate_sphere)
                candidates_k_list.append(candidate_k)

            candidates_sphere = torch.cat(candidates_sphere_list, dim=0)
            candidates_k = torch.cat(candidates_k_list, dim=0)

            print("All embeddings loaded. Calculating angles...")

            q_thetas, q_psi1s, q_psi2s = cartesian_to_spherical_angles(
                q_sphere)
            c_thetas, c_psi1s, c_psi2s = cartesian_to_spherical_angles(
                candidates_sphere)

            all_data = {
                'query_theta': q_thetas.cpu().numpy(),
                'query_psi1': q_psi1s.cpu().numpy(),
                'query_psi2': q_psi2s.cpu().numpy(),
                'query_k': q_k.cpu().numpy(),
                'candidate_theta': c_thetas.cpu().numpy(),
                'candidate_psi1': c_psi1s.cpu().numpy(),
                'candidate_psi2': c_psi2s.cpu().numpy(),
                'candidate_k': candidates_k.cpu().numpy(),
            }

            print("Generating combined distribution plots...")

            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            fig.suptitle(
                'Global Distributions of Query vs. Candidate Angles and K-Values', fontsize=20)

            plot_config = {'kde': True, 'bins': 50}

            sns.histplot(all_data['query_theta'], ax=axes[0],
                         **plot_config, color='royalblue', label='Queries')
            sns.histplot(all_data['candidate_theta'], ax=axes[0],
                         **plot_config, color='darkorange', label='Candidates')
            axes[0].set_title("Longitudinal Angle (θ)", fontsize=14)
            axes[0].set_xlim(0, 2 * np.pi)

            sns.histplot(all_data['query_psi1'], ax=axes[1], **
                         plot_config, color='royalblue', label='Queries')
            sns.histplot(all_data['candidate_psi1'], ax=axes[1], **
                         plot_config, color='darkorange', label='Candidates')
            axes[1].set_title("1st Latitudinal Angle (ψ1)", fontsize=14)
            axes[1].set_xlim(0, np.pi)

            sns.histplot(all_data['query_psi2'], ax=axes[2], **
                         plot_config, color='royalblue', label='Queries')
            sns.histplot(all_data['candidate_psi2'], ax=axes[2], **
                         plot_config, color='darkorange', label='Candidates')
            axes[2].set_title("2nd Latitudinal Angle (ψ2)", fontsize=14)
            axes[2].set_xlim(0, np.pi)

            sns.histplot(all_data['query_k'], ax=axes[3], **
                         plot_config, color='royalblue', label='Queries')
            sns.histplot(all_data['candidate_k'], ax=axes[3], **
                         plot_config, color='darkorange', label='Candidates')
            axes[3].set_title("K-Value (Concentration)", fontsize=14)

            for i in range(4):
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                if i < 3:
                    axes[i].set_xlabel('Angle (radians)')
                else:
                    axes[i].set_xlabel('K Value')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = "global_angle_and_k_distributions.png"
            plt.savefig(save_path)
            plt.show()

            print(f"Plots saved to {save_path}")
