import os
import time
import numpy as np
import pickle as pkl
import torch
import sys
import json
from tqdm import tqdm
from torch import optim
from transformers import BertTokenizer, AutoTokenizer
from utils import *
from data import *
from model import PolarTaxo
import matplotlib.pyplot as plt
from optimizer import RiemannianAdam
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import wandb

os.environ["WANDB_MODE"] = "online"


class Experiments(object):

    def __init__(self, args):
        super(Experiments, self).__init__()

        self.args = args
        self.tokenizer = self.__load_tokenizer__()
        self.train_loader, self.train_set = load_data(
            self.args, self.tokenizer, "train")
        self.test_loader, self.test_set = load_data(
            self.args, self.tokenizer, "test")
        self.accumulation_steps = self.args.accumulation_steps

        self.model = PolarTaxo(args)

        self.optimizer = self._select_optimizer()
        self._set_device()
        self.exp_setting = "_".join([str(elem) for elem in [self.args.pre_train, self.args.dataset, self.args.expID, self.args.epochs,
                                    self.args.batch_size, self.args.beta, self.args.embed_size, self.args.geometric_weight, self.args.probabilistic_weight]])

        setting = {
            "dataset": self.args.dataset,
            "expID": self.args.expID,
            'beta': self.args.beta,
            'embed_size': self.args.embed_size,
            'seed': self.args.seed,
            'c': self.args.c
        }
        print(setting)

    def __load_tokenizer__(self):
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
            # optimizer = optim.AdamW(
            #     parameters, lr=self.args.lr, eps=self.args.eps)
            optimizer = RiemannianAdam(
                params=parameters, lr=self.args.lr, eps=self.args.eps, betas=(0.9, 0.999))

        return optimizer

    def _set_device(self):
        if self.args.cuda:
            self.model = self.model.cuda()

    def train_one_step(self, it, encode_parent, encode_child, encode_negative):

        self.model.train()

        loss = self.model(
            it, encode_parent, encode_child, encode_negative)

        loss.backward()

        self.optimizer.step()

        self.model.normalize_spherical_weights()
        self.optimizer.zero_grad()

        del encode_parent, encode_child, encode_negative

        torch.cuda.empty_cache()
        gc.collect()

        return loss

    def train(self, checkpoint=None, save_path=None):
        time_tracker = []
        # save_path = f"../final_result/{self.args.dataset}/KL_volume_containment_{self.args.expID}_{self.args.method}_{self.args.model}_{self.args.negsamples}.pt"
        test_acc = test_mrr = test_wu_p = 0
        old_test_acc = old_test_mrr = old_test_wu_p = 0

        if checkpoint:
            self.model.load_state_dict(torch.load(f"{checkpoint}"))

        if save_path is None:
            savedir = os.path.join("../result", self.args.dataset, "model")
            traindir = os.path.join("../result", self.args.dataset, "train")
            if not os.path.exists(savedir):
                os.makedirs(savedir, exist_ok=True)
            if not os.path.exists(traindir):
                os.makedirs(traindir, exist_ok=True)
            save_path = os.path.join(
                "../result", self.args.dataset, "model", f"exp_model_{self.exp_setting}.checkpoint")

        for epoch in tqdm(range(self.args.epochs)):
            epoch_time = time.time()
            train_loss = []
            theta_train_loss = []
            psi_train_loss = []

            self.optimizer.zero_grad()
            for i, (encode_parent, encode_child, encode_negative) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                loss = self.train_one_step(
                    it=i, encode_parent=encode_parent, encode_child=encode_child, encode_negative=encode_negative)

                train_loss.append(loss.item())

            # train_loss = np.average(train_loss)
            # theta_train_loss = np.average(theta_train_loss)
            # psi_train_loss = np.average(psi_train_loss)
            train_loss = np.average(train_loss)
            print("Theta Loss: ", theta_train_loss)
            print("Psi Loss: ", psi_train_loss)
            print("Loss: ", train_loss)

            # torch.save(self.model.state_dict(), os.path.join(
            #     "../result", self.args.dataset, "train", f"KL_volume_containment_{self.args.expID}_{self.args.method}_{self.args.model}_{self.args.negsamples}_{epoch}.checkpoint"))
            # if epoch >= 1:
            #     os.remove(os.path.join(
            #         "../result", self.args.dataset, "train", f"KL_volume_containment_{self.args.expID}_{self.args.method}_{self.args.model}_{self.args.negsamples}_{epoch-1}.checkpoint"))

            test_metrics = self.predict()
            test_acc = test_metrics["Prec@1"]
            test_mrr = test_metrics["MRR"]
            # test_wu_p = test_metrics["Wu"]
            if test_acc >= old_test_acc or test_mrr >= old_test_mrr:
                torch.save(self.model.state_dict(
                ), f"../final_result/{self.args.dataset}/experiment_{self.exp_setting}.pt")
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
            # Save the checkpoint
            torch.save(self.model.state_dict(
            ), f"../result/{self.args.dataset}/train/experiment_{self.exp_setting}.checkpoint")

            # torch.save(self.model.state_dict(), os.path.join("../result", self.args.dataset,
            #            "train", "exp_model_"+self.exp_setting+"_"+str(epoch)+".checkpoint"))
            # if epoch:
            #     os.remove(os.path.join("../result", self.args.dataset, "train",
            #               "exp_model_"+self.exp_setting+"_"+str((epoch-1))+".checkpoint"))

    def get_pos_from_h_theta(self, h, theta):

        r = self.args.r_0 * torch.exp(h)

        theta_norm = torch.norm(theta, p=2, dim=1, keepdim=True)
        theta_unit = theta / (theta_norm + 1e-8)

        pos = r * theta_unit
        return pos, r, h

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
                # probabilistic_score = -torch.sum(vmf_kl_divergence(
                #     q_mu, q_k, candidates_mu, candidates_k, candidates_mu.size(1)), dim=1)

                # TODO: Figure this combination out.
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

        with open(f'../results/{self.args.dataset}/res_{self.exp_setting}.json', 'a+') as f:
            d = vars(self.args)
            expt_details = {
                "Arguments": d,
                "Test Metrics": test_metrics,
            }
            json.dump(expt_details, f, indent=4)

        return test_metrics

    def visualize_angle_distributions(self, tag=None, path=None):
        print("Prediction starting.....")
        store_csv = False
        if tag == "test" and path:
            self.model.load_state_dict(torch.load(path))
            store_csv = True

        self.model.eval()
        with torch.no_grad():
            q_sphere = self.model.child_projection(
                self.test_set.encode_query)
            candidates_sphere = list()

            for encode_candidate in self.test_loader:
                candidate_sphere = self.model.par_projection(
                    encode_candidate)

                candidates_sphere.append(candidate_sphere)

            candidates_sphere = torch.cat(candidates_sphere, dim=0)
            num_queries = q_sphere.size(0)

            if store_csv is True:
                all_query_thetas, all_query_psi1s, all_query_psi2s = [], [], []
                all_top1_pred_thetas, all_top1_pred_psi1s, all_top1_pred_psi2s = [], [], []

                with open('angle_distributions.csv', 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    header = [
                        'query_theta', 'query_psi1', 'query_psi2',
                        'pred1_theta', 'pred1_psi1', 'pred1_psi2',
                        'pred2_theta', 'pred2_psi1', 'pred2_psi2',
                        'pred3_theta', 'pred3_psi1', 'pred3_psi2'
                    ]

                    csv_writer.writerow(header)

                    for i in tqdm(range(num_queries), desc='Evaluating Queries'):
                        q_sph = q_sphere[i].unsqueeze(0)
                        q_sph_expanded = q_sph.expand(
                            candidates_sphere.shape[0], -1)
                        geometric_score = torch.sum(
                            candidates_sphere * q_sph_expanded, dim=1)
                        final_score = geometric_score

                        _, top_indices = torch.topk(final_score, 3)
                        top_3_candidates_sph = candidates_sphere[top_indices]

                        q_theta, q_psi1, q_psi2 = cartesian_to_spherical_angles(
                            q_sph)

                        pred_thetas, pred_psi1s, pred_psi2s = cartesian_to_spherical_angles(
                            top_3_candidates_sph)
                        row_data = [
                            q_theta.item(), q_psi1.item(), q_psi2.item(),
                            pred_thetas[0].item(), pred_psi1s[0].item(
                            ), pred_psi2s[0].item(),
                            pred_thetas[1].item(), pred_psi1s[1].item(
                            ), pred_psi2s[1].item(),
                            pred_thetas[2].item(), pred_psi1s[2].item(
                            ), pred_psi2s[2].item()
                        ]
                        csv_writer.writerow(row_data)

                        all_query_thetas.append(q_theta.item())
                        all_query_psi1s.append(q_psi1.item())
                        all_query_psi2s.append(q_psi2.item())
                        all_top1_pred_thetas.append(pred_thetas[0].item())
                        all_top1_pred_psi1s.append(pred_psi1s[0].item())
                        all_top1_pred_psi2s.append(pred_psi2s[0].item())

                    print("Generating angle distribution plots...")

                    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
                    fig.suptitle(
                        'Distributions of Query vs. Top-1 Predicted Angles', fontsize=20)

                    plot_config = {
                        'kde': True,
                        'bins': 50
                    }

                    sns.histplot(all_query_thetas, ax=axes[0, 0], **plot_config, color='royalblue').set_title(
                        "Query: Longtiudinal Angle", fontsize=14)
                    sns.histplot(all_query_psi1s, ax=axes[0, 1], **plot_config, color='royalblue').set_title(
                        "Query: First Latitudinal Angle", fontsize=14)
                    sns.histplot(all_query_psi2s, ax=axes[0, 2], **plot_config, color='royalblue').set_title(
                        "Query: Second Latitudinal Angle", fontsize=14)

                    sns.histplot(all_top1_pred_thetas, ax=axes[1, 0], **plot_config, color='darkorange').set_title(
                        'Top-1 Pred: Longitudinal Angle (θ)', fontsize=14)
                    sns.histplot(all_top1_pred_psi1s, ax=axes[1, 1], **plot_config, color='darkorange').set_title(
                        'Top-1 Pred: 1st Latitudinal Angle (ψ1)', fontsize=14)
                    sns.histplot(all_top1_pred_psi2s, ax=axes[1, 2], **plot_config, color='darkorange').set_title(
                        'Top-1 Pred: 2nd Latitudinal Angle (ψ2)', fontsize=14)

                    for i in range(3):
                        axes[0, i].set_xlabel('Angle (radians)')
                        axes[1, i].set_xlabel('Angle (radians)')
                        axes[0, i].set_ylabel('Frequency')
                        axes[1, i].set_ylabel('Frequency')

                    axes[0, 0].set_xlim(0, 2 * np.pi)
                    axes[1, 0].set_xlim(0, 2 * np.pi)
                    axes[0, 1].set_xlim(0, np.pi)
                    axes[1, 1].set_xlim(0, np.pi)
                    axes[0, 2].set_xlim(0, np.pi)
                    axes[1, 2].set_xlim(0, np.pi)

                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.savefig("angle_distribution_plots.png")
