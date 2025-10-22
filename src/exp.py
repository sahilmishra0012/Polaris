
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
import wandb
from vmf import VMFRegularisation

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
                                    self.args.batch_size, self.args.beta, self.args.embed_size, self.args.kappa]])

        setting = {
            "dataset": self.args.dataset,
            "expID": self.args.expID,
            'beta': self.args.beta,
            'embed_size': self.args.embed_size,
            'seed': self.args.seed,
            'c': self.args.c
        }
        print(setting)

        # if self.args.wandb:
        #     wandb.init(project='gaussian', config=setting, entity='taxo_iitd')
        #     wandb.run.log_code(".")

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

            if self.args.is_multi_parent is True:
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
            else:
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
            ), f"../result/{self.args.dataset}/train/experiment_{self.exp_setting}.")

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
        if tag == "test" and path:
            self.model.load_state_dict(torch.load(path))

        self.model.eval()
        with torch.no_grad():
            score_list = []
            gt_label = self.test_set.test_gt_id

            q_sphere = self.model.child_projection(
                self.test_set.encode_query)

            # q_angles = self.model.to_polar(q_sphere)
            # q_theta, q_psi = q_angles[:, :-
            #                           1], q_angles[:, -1].view(q_angles.size(0))

            candidate_thetas = list()
            candidate_psis = list()
            candidates_sphere = list()
            for encode_candidate in self.test_loader:
                candidate_sphere = self.model.par_projection(
                    encode_candidate)
                # candidate_angles = self.model.to_polar(candidate_sphere)
                # candidate_theta, candidate_psi = candidate_angles[:,
                #                                                   :-1], candidate_angles[:, -1].view(candidate_angles.size(0))

                candidates_sphere.append(candidate_sphere)
                # candidate_psis.append(candidate_psi)
                # candidate_thetas.append(candidate_theta)

            # candidates_thetas = torch.cat(candidate_thetas, dim=0)
            # candidates_psis = torch.cat(candidate_psis, dim=0)
            candidates_sphere = torch.cat(candidates_sphere, dim=0)

            num_queries = q_sphere.size(0)
            num_candidates = candidates_sphere.size(0)

            for i in tqdm(range(num_queries), desc='Evaluating Queries'):
                # q_the = q_theta[i].unsqueeze(0).expand(num_candidates, -1)
                # q_ps = q_psi[i].unsqueeze(0).expand(num_candidates, -1)
                q_sph = q_sphere[i].unsqueeze(0).expand(num_candidates, -1)

                # distance = q_the*candidates_thetas
                # lat_distance = self.model.latitude_distance(
                #     candidates_thetas, q_the)
                # long_distance = self.model.longitude_distance(
                #     candidates_psis, q_ps)

                # lat_distance = self.model.latitude_distance(
                #     candidates_psis, q_ps)
                final_score = torch.sum(candidates_sphere*q_sph, dim=1)
                # final_score = torch.mean(
                #     self.args.lat_weight*lat_distance+self.args.long_weight*long_distance, dim=0)

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
