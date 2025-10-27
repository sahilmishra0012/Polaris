import numpy as np
import pandas as pd
import random
import torch
import pytz
from datetime import datetime, timezone
import math
import matplotlib.pyplot as plt


def cartesian_to_spherical_angles(cartesian_vectors):
    if cartesian_vectors.shape[1] < 3:
        raise ValueError(
            "Input dimension must be at least 3 to compute the required angles.")

    epsilon = 1e-9

    xn = cartesian_vectors[:, -1]
    xn_minus_1 = cartesian_vectors[:, -2]

    numerator_theta = xn_minus_1 + torch.sqrt(xn**2 + xn_minus_1**2 + epsilon)
    argument_theta = numerator_theta / (xn + epsilon)

    arccot_val = np.pi / 2 - torch.atan(argument_theta)
    theta = 2 * arccot_val

    phi1 = torch.acos(torch.clamp(cartesian_vectors[:, 0], -1.0, 1.0))

    denominator_phi2 = torch.sqrt(1.0 - cartesian_vectors[:, 0]**2 + epsilon)
    argument_phi2 = cartesian_vectors[:, 1] / denominator_phi2
    phi2 = torch.acos(torch.clamp(argument_phi2, -1.0, 1.0))

    return theta, phi1, phi2


def get_long_angle(mu):

    return np.arctan2(mu[-1], mu[-1])


def visualize_angles(mu_q, mu_p, mu_t1, mu_t2, fname):
    angle_q = get_long_angle(mu_q)
    angle_p = get_long_angle(mu_p)
    angle_t1 = get_long_angle(mu_t1)
    angle_t2 = get_long_angle(mu_t2)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    points_to_plot = [
        {'angle': angle_q, 'radius': 1, 'label': 'Query',
            'color': 'blue', 'marker': '*', 's': 200},
        {'angle': angle_p, 'radius': 1, 'label': 'Predicted Parent',
            'color': 'green', 'marker': 'o', 's': 150},
        {'angle': angle_t1, 'radius': 1, 'label': 'Top Candidate 1',
            'color': 'red', 'marker': 's', 's': 100},
        {'angle': angle_t2, 'radius': 1, 'label': 'Top Candidate 2',
            'color': 'purple', 'marker': 'D', 's': 100}
    ]

    for point in points_to_plot:
        if point['label'] == 'Top Candidate 1' and np.isclose(point['angle'], angle_p):
            point['marker'] = 'x'
            point['s'] = 200
            point['label'] = 'Top Candidate 1 (Predicted)'

        ax.scatter(point['angle'], point['radius'], c=point['color'],
                   marker=point['marker'], s=point['s'], label=point['label'], zorder=5)

        ax.set_title('Polar Coordinates Visualisation', fontsize=16, pad=20)

        ax.set_yticklabels([])

        ax.set_rlim(0, 1.3)

        ax.grid(True, zorder=0)

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Visualization saved to {fname}")


def bert_embedding_to_spherical(e):
    norm = torch.norm(e, p=2, dim=1, keepdim=True)
    unit_vectors = e / (norm + 1e-9)

    batch_size, d = unit_vectors.shape
    num_angles = d - 1
    angles = torch.zeros(batch_size, num_angles, device=e.device)

    cumulative_sum_sq = torch.zeros(batch_size, device=e.device)

    for i in range(d - 1, 0, -1):
        x_i = unit_vectors[:, i]
        cumulative_sum_sq = cumulative_sum_sq + x_i**2
        denominator = torch.sqrt(cumulative_sum_sq)

        ratio = unit_vectors[:, i-1] / (denominator + 1e-9)
        clamped_ratio = torch.clamp(ratio, -1.0, 1.0)

        angles[:, i-1] = torch.acos(clamped_ratio)

    x_last = unit_vectors[:, -1]
    x_second_last = unit_vectors[:, -2]

    theta_denom = torch.sqrt(x_second_last**2 + x_last**2) + 1e-9
    theta_ratio = x_second_last / theta_denom
    theta_base = torch.acos(torch.clamp(theta_ratio, -1.0, 1.0))

    final_theta = torch.where(x_last < 0, 2 * math.pi - theta_base, theta_base)

    angles[:, -1] = final_theta

    theta_out = angles[:, -1].unsqueeze(1)
    psi_out = angles[:, :-1]

    return theta_out, psi_out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_local_time():
    utc_dt = datetime.now(timezone.utc)
    PST = pytz.timezone('Asia/Kolkata')
    print("Pacific time {}".format(utc_dt.astimezone(PST).isoformat()))
    return


def accuracy(pred, gt, tr, te):
    preds = np.array(list(pred[:, 0]))
    gts = np.array(list(gt))
    acc = np.sum(preds == gts)/len(gt)
    print("Predictions: ", preds)
    print("GT: ", gts)
    for i in range(len(preds)):
        print(f"Predicted: {tr[preds[i]]}, GT: {te[gts[i]]}")
    return acc


def precision_k(pred, gt, k):
    preds = np.array(list(pred[:, :k]))
    gts = np.array(list(gt))
    val = np.sum(preds == gts[:, np.newaxis])*1.0/(len(gt)*k)
    return val


def hits_at_k_multi_p(pred, gt, k):
    num_queries = len(gt)
    if num_queries == 0:
        return 0.0

    num_hits = 0
    for i in range(num_queries):

        preds_k = set(pred[i][:k])
        true_parents = set(gt[i])

        if not true_parents:

            continue

        if not preds_k.isdisjoint(true_parents):
            num_hits += 1

    return num_hits / num_queries


def recall_k(pred, gt, k):
    num_queries = len(gt)
    if num_queries == 0:
        return 0.0

    preds_k = pred[:, :k]
    gts = np.array(gt)[:, np.newaxis]

    hits_matrix = (preds_k == gts)
    successful_queries = np.any(hits_matrix, axis=1)

    return np.mean(successful_queries)


def rank_scores(pred, gt):
    mrr = 0
    mr = 0
    dcg = 0.0
    idcg = 0.0
    cnt = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == gt[i]:
                mr += (j+1)
                mrr += (1/(j+1))
                cnt += 1
                dcg += (1/np.log2((j+1)+1))
                idcg += (1/(np.log2(cnt+1)))
                break
    ndcg = dcg/idcg if idcg != 0 else 0
    ndcg = ndcg/len(gt)
    mrr = mrr/len(gt)
    mr = mr/len(gt)

    return mrr, mr, ndcg


def rank_scores_multi_p(pred, gt):
    total_mrr = 0.0
    total_mr = 0.0
    num_triplets = 0

    for i in range(len(gt)):
        ranked_list = pred[i]

        true_parents = gt[i]

        num_triplets += len(true_parents)

        rank_map = {item: rank + 1 for rank, item in enumerate(ranked_list)}

        for true_parent in true_parents:
            if true_parent in rank_map:
                rank = rank_map[true_parent]

                total_mr += rank
                total_mrr += 1 / rank
            else:
                pass

    if num_triplets == 0:
        return 0.0, 0.0

    mrr = total_mrr / num_triplets
    mr = total_mr / num_triplets

    return mrr, mr


def mrr_score(pred, gt):
    mrr = 0
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == gt[i]:
                mrr += 1/(j+1)
    mrr = mrr/len(gt)
    return mrr


def wu_p_score(pred, gt, path2root, compiled):

    pred = np.squeeze(pred[:, 0])
    wu_p = 0
    for i in range(len(pred)):
        path_pred = path2root[pred[i]]
        path_gt = path2root[gt[i]]
        compiled[i].append(len(path_gt)-1)
        shared_nodes = set(path_pred) & set(path_gt)
        lca_depth = 1
        for node in shared_nodes:
            lca_depth = max(len(path2root[node])-1, lca_depth)
        wu_p += 2*lca_depth/(len(path_pred)+len(path_gt))

    wu_p = wu_p/len(gt)

    return wu_p


def recall_k_multi_p(pred, gt, k):

    total_hits = 0
    num_triplets = 0

    for i in range(len(gt)):
        preds_k = set(pred[i][:k])

        true_parents = gt[i]
        num_triplets += len(true_parents)

        if not true_parents:
            continue

        for true_parent in true_parents:
            if true_parent in preds_k:
                total_hits += 1

    if num_triplets == 0:
        return 0.0

    return total_hits / num_triplets


def wu_p_score(pred, gt, path2root, compiled):

    pred = np.squeeze(pred[:, 0])
    wu_p = 0
    for i in range(len(pred)):
        path_pred = path2root[pred[i]]
        path_gt = path2root[gt[i]]
        compiled[i].append(len(path_gt))
        shared_nodes = set(path_pred) & set(path_gt)
        lca_depth = 1
        for node in shared_nodes:
            lca_depth = max(len(path2root[node]), lca_depth)
        wu_p += 2*lca_depth/(len(path_pred)+len(path_gt))

    wu_p = wu_p/len(gt)

    return wu_p


def metrics_multi_p(indices, gt, candidate_list, id_concept, test_concepts_id):
    ind = np.squeeze(indices.detach().cpu().numpy())
    x, y = ind.shape

    pred = np.zeros_like(ind)

    for i in range(x):

        pred[i] = candidate_list[ind[i]]

    mrr, mr = rank_scores_multi_p(
        pred, gt)
    prec5 = hits_at_k_multi_p(pred, gt, 5)
    prec10 = hits_at_k_multi_p(pred, gt, 10)
    prec1 = hits_at_k_multi_p(pred, gt, 1)
    rec1 = recall_k_multi_p(pred, gt, 1)
    rec5 = recall_k_multi_p(pred, gt, 5)
    rec10 = recall_k_multi_p(pred, gt, 10)

    return {"Prec@1": prec1, "MRR": mrr, "MR": mr, "Recall@1": rec1, "Prec@5": prec5, "Prec@10": prec10, "Recall@5": rec5, "Recall@10": rec10}


def metrics(indices, gt, train_concept_set, path2root, testid2concept, trainid2concept, testconcepts, sortedscores):
    ind = np.squeeze(indices.detach().cpu().numpy())
    x, y = ind.shape
    pred = np.array([[i for i in range(y)] for _ in range(x)])
    compiled = [[testid2concept[testconcepts[i]], trainid2concept[gt[i]],
                 sortedscores[i][0].item()] for i in range(x)]

    for i in range(len(pred)):
        pred[i] = np.array(list(train_concept_set))[ind[i]]
        compiled[i].append(trainid2concept[pred[i][0]])
        compiled[i].append(True if pred[i][0] == gt[i] else False)

    acc = accuracy(pred, gt, trainid2concept, testid2concept)
    mrr, mr, ndcg = rank_scores(pred, gt)
    wu_p = wu_p_score(pred, gt, path2root, compiled)
    prec5 = precision_k(pred, gt, 5)
    prec10 = precision_k(pred, gt, 10)
    prec1 = precision_k(pred, gt, 1)
    rec1 = recall_k(pred, gt, 1)
    rec10 = recall_k(pred, gt, 10)
    rec5 = recall_k(pred, gt, 5)
    depth = [elem[-2:] for elem in compiled]
    np.savetxt("depth_analysis.csv", depth, fmt="%s,%i", delimiter=",")

    file_path = 'depth_analysis.csv'
    data = pd.read_csv(file_path, header=None, names=['correct', 'depth'])

    report = data.groupby('depth').agg(
        total_attempts=('correct', 'count'),
        correct_answers=('correct', 'sum'),
        accuracy=('correct', lambda x: x.mean() * 100)
    ).reset_index()

    report.columns = ['Depth', 'Total Attempts',
                      'Correct Answers', 'Accuracy (%)']

    print(report)
    return {"Prec@1": prec1, "MRR": mrr, "MR": mr, "Wu": wu_p, "Prec@5": prec5, "Prec@10": prec10, "Recall@1": rec1, "Recall@5": rec5, "Recall@10": rec10}
