import numpy as np
import pandas as pd
import random
import torch
import pytz
from datetime import datetime, timezone
import math
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import plotly.graph_objects as go


def plot_concept_space_map(query_info, predicted_info, ground_truth_info, save_path):

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    all_concepts = [query_info, predicted_info] + ground_truth_info

    unique_concepts = []
    seen_names = set()
    for concept in all_concepts:
        if concept['name'] not in seen_names:
            unique_concepts.append(concept)
            seen_names.add(concept['name'])

    concept_coords = {}
    for concept in unique_concepts:
        x_embed = concept['embedding'][0]
        y_embed = concept['embedding'][1]

        theta = np.arctan2(y_embed, x_embed)

        conceptual_radius = concept['radius']

        plot_x = conceptual_radius * np.cos(theta)
        plot_y = conceptual_radius * np.sin(theta)
        concept_coords[concept['name']] = (plot_x, plot_y)

    plotted_radii = set()
    for concept in unique_concepts:
        r = concept['radius']
        if r in plotted_radii or r < 1e-6:
            continue
        orbit = plt.Circle((0, 0), r, color='gray', linestyle='--',
                           linewidth=0.8, fill=False, alpha=0.6)
        ax.add_patch(orbit)
        plotted_radii.add(r)

    for concept in unique_concepts:
        x, y = concept_coords[concept['name']]
        name = concept['name']
        radius_text = f"\n(R: {concept['radius']:.2f})"

        if name == query_info['name']:
            ax.scatter(x, y, s=400, c='gold', marker='*',
                       zorder=10, label=f"Query")
            ax.text(x, y, f" {name}{radius_text}", ha='center',
                    va='top', fontsize=10, color='gold', fontweight='bold')
        elif name == predicted_info['name']:
            ax.scatter(x, y, s=200, c='deepskyblue', marker='o',
                       zorder=10, label=f"Prediction")
            ax.text(x, y, f" {name}{radius_text}", ha='center',
                    va='top', fontsize=9, color='deepskyblue')
        else:
            ax.scatter(x, y, s=250, c='limegreen', marker='D',
                       zorder=10, label=f"Ground Truth")
            ax.text(x, y, f" {name}{radius_text}", ha='center',
                    va='top', fontsize=9, color='limegreen')

    ax.legend(loc='upper right')
    ax.set_title(
        f"Concept Space Map for Query '{query_info['name']}'", fontsize=14, pad=20)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    max_r = max([c['radius'] for c in all_concepts if 'radius' in c] + [0])
    ax.set_xlim(-max_r * 1.3, max_r * 1.3)
    ax.set_ylim(-max_r * 1.3, max_r * 1.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_static_solar_system(query_info, candidates_info, ground_truth_info, save_path):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    all_concepts = [query_info] + candidates_info + ground_truth_info

    unique_concepts = []
    seen_names = set()
    for concept in all_concepts:
        if concept['name'] not in seen_names:
            unique_concepts.append(concept)
            seen_names.add(concept['name'])

    embeddings = np.array([c['embedding'] for c in unique_concepts])

    if len(embeddings) < 2:
        print(
            f"Warning: Not enough unique points ({len(embeddings)}) for 2D PCA. Skipping plot.")
        return

    pca = PCA(n_components=2)
    low_dim_embeddings = pca.fit_transform(embeddings)

    concept_coords = {}
    for i, concept in enumerate(unique_concepts):
        vec = low_dim_embeddings[i]
        conceptual_radius = concept['radius']

        theta = np.arctan2(vec[1], vec[0])

        plot_x = conceptual_radius * np.cos(theta)
        plot_y = conceptual_radius * np.sin(theta)
        concept_coords[concept['name']] = (plot_x, plot_y)

    plotted_radii = set()
    for concept in candidates_info + ground_truth_info:
        r = concept['radius']
        if r in plotted_radii or r == 0:
            continue

        orbit = plt.Circle((0, 0), r, color='gray', linestyle='--',
                           linewidth=0.8, fill=False, alpha=0.6)
        ax.add_patch(orbit)
        plotted_radii.add(r)

    ax.scatter(0, 0, s=400, c='gold', marker='*',
               zorder=10, label='Query (Sun)')
    ax.text(0, 0, f" {query_info['name']}\n (R: {query_info['radius']:.2f})",
            ha='center', va='top', fontsize=9, color='white', fontweight='bold')

    def plot_planets(concepts, color, marker, label):
        for concept in concepts:
            x, y = concept_coords[concept['name']]
            ax.scatter(x, y, s=150, c=color, marker=marker,
                       zorder=10, label=label)
            ax.text(x, y, f" {concept['name']}\n (R: {concept['radius']:.2f})",
                    ha='center', va='top', fontsize=9, color='white')

    plot_planets(candidates_info, color='deepskyblue',
                 marker='o', label='Predicted')
    plot_planets(ground_truth_info, color='limegreen',
                 marker='D', label='Ground Truth')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

    ax.set_title(
        f"Concept Solar System for Query: '{query_info['name']}'", fontsize=16, pad=20)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    max_r = max([c['radius'] for c in all_concepts if 'radius' in c] + [0])
    ax.set_xlim(-max_r * 1.2, max_r * 1.2)
    ax.set_ylim(-max_r * 1.2, max_r * 1.2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_interactive_solar_system(query_info, candidates_info, ground_truth_info, save_path, dim='3d'):

    all_concepts = [query_info] + candidates_info + ground_truth_info

    unique_concepts = []
    seen_names = set()
    for concept in all_concepts:
        if concept['name'] not in seen_names:
            unique_concepts.append(concept)
            seen_names.add(concept['name'])

    embeddings = np.array([c['embedding'] for c in unique_concepts])
    n_components = 3 if dim == '3d' else 2

    if len(embeddings) < n_components:
        print(
            f"Warning: Not enough unique points ({len(embeddings)}) for {n_components}D PCA. Skipping plot.")
        return

    pca = PCA(n_components=n_components)
    low_dim_embeddings = pca.fit_transform(embeddings)

    concept_coords = {}
    for i, concept in enumerate(unique_concepts):
        vec = low_dim_embeddings[i]
        radius = concept['radius']

        if dim == '3d':
            x, y, z = vec
            theta = np.arctan2(y, x)
            phi = np.arccos(z / (np.linalg.norm(vec) + 1e-8))
            plot_x = radius * np.sin(phi) * np.cos(theta)
            plot_y = radius * np.sin(phi) * np.sin(theta)
            plot_z = radius * np.cos(phi)
        else:
            x, y = vec
            theta = np.arctan2(y, x)
            plot_x = radius * np.cos(theta)
            plot_y = radius * np.sin(theta)
            plot_z = 0

        concept_coords[concept['name']] = (plot_x, plot_y, plot_z)

    fig = go.Figure()

    def add_concept_trace(concepts, name, symbol, color, size):
        for concept in concepts:
            x, y, z = concept_coords[concept['name']]
            hover_text = f"<b>{concept['name']}</b><br>Type: {name}<br>Radius: {concept['radius']:.3f}"

            plot_func = go.Scatter3d if dim == '3d' else go.Scatter
            fig.add_trace(plot_func(
                x=[x], y=[y], z=[z] if dim == '3d' else None,
                mode='markers',
                marker=dict(symbol=symbol, color=color, size=size),
                name=name,
                hoverinfo='text',
                text=[hover_text]
            ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(symbol='circle', color='gold', size=14),
        name='Query (Sun)',
        hoverinfo='text',
        text=[
            f"<b>{query_info['name']}</b><br>Type: Query<br>Radius: {query_info['radius']:.3f}"]
    ))

    add_concept_trace(candidates_info, 'Predicted', 'circle', 'deepskyblue', 8)
    add_concept_trace(ground_truth_info, 'Ground Truth',
                      'diamond', 'limegreen', 12)

    plotted_radii = set()
    for concept in candidates_info + ground_truth_info:
        r = concept['radius']
        if r in plotted_radii or r == 0:
            continue

        if dim == '3d':
            theta_orb = np.linspace(0, 2 * np.pi, 100)
            x_orb = r * np.cos(theta_orb)
            y_orb = r * np.sin(theta_orb)
            z_orb = np.zeros_like(x_orb)
            fig.add_trace(go.Scatter3d(x=x_orb, y=y_orb, z=z_orb, mode='lines',
                                       line=dict(color='white',
                                                 width=1, dash='dot'),
                                       name=f'Orbit (r={r:.2f})', hoverinfo='none'))
        else:
            theta_orb = np.linspace(0, 2 * np.pi, 100)
            x_orb = r * np.cos(theta_orb)
            y_orb = r * np.sin(theta_orb)
            fig.add_trace(go.Scatter(x=x_orb, y=y_orb, mode='lines',
                                     line=dict(color='gray',
                                               width=1, dash='dot'),
                                     name=f'Orbit (r={r:.2f})', hoverinfo='none'))
        plotted_radii.add(r)

    title = f"Interactive Solar System for Query: '{query_info['name']}'"
    if dim == '3d':
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='black',
                aspectmode='cube'
            ),
            showlegend=True,
            legend_title_text='Concept Type'
        )
    else:
        fig.update_layout(
            title=title,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='black',
            showlegend=True,
            legend_title_text='Concept Type',
            yaxis_scaleanchor="x"
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)


def plot_radii_comparison(query_info, top_candidates_info, save_path, principle='traditional'):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    all_concepts = [query_info] + top_candidates_info
    all_radii = [c['radius'] for c in all_concepts]
    max_radius = max(all_radii) if all_radii else 1.0

    colors = plt.cm.viridis(np.linspace(0, 1, 5))

    query_circle = plt.Circle((0, 0), query_info['radius'], color=colors[0], alpha=0.4,
                              label=f"Query: {query_info['name']}")
    ax.add_patch(query_circle)

    for i, candidate in enumerate(top_candidates_info):
        candidate_circle = plt.Circle((0, 0), candidate['radius'], color=colors[i + 1], alpha=0.4,
                                      label=f"Top {i+1}: {candidate['name']}")
        ax.add_patch(candidate_circle)

    angle_step = 65

    angle = np.deg2rad(10)
    text_radius = query_info['radius'] + 0.05 * max_radius
    x_q = text_radius * np.cos(angle)
    y_q = text_radius * np.sin(angle)
    ax.text(x_q, y_q, f"Query\n(R: {query_info['radius']:.2f})",
            ha='center', va='center', fontweight='bold', fontsize=10, color=colors[0])

    for i, candidate in enumerate(top_candidates_info):
        angle = np.deg2rad(10 + (i + 1) * angle_step)
        text_radius = candidate['radius'] + 0.05 * max_radius
        x_c = text_radius * np.cos(angle)
        y_c = text_radius * np.sin(angle)
        ax.text(x_c, y_c, f"Top {i+1}\n(R: {candidate['radius']:.2f})",
                ha='center', va='center', fontsize=9, color=colors[i+1])

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-max_radius * 1.3, max_radius * 1.3)
    ax.set_ylim(-max_radius * 1.3, max_radius * 1.3)
    ax.axis('off')

    title_principle = "Parent > Child" if principle == 'traditional' else "Child > Parent"
    plt.title(f"Radius Comparison for Query: \"{query_info['name']}\"\n(Principle: {title_principle})",
              fontweight='bold')

    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def cartesian_to_spherical_angles(e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, d = e.shape

    if d < 4:
        raise ValueError(
            f"Input dimension must be at least 4 to return theta, psi1, and psi2, but got d={d}.")

    eps = 1e-8

    e_sq = e.pow(2)
    cum_sq_from_back = torch.flip(torch.cumsum(
        torch.flip(e_sq, dims=[1]), dim=1), dims=[1])

    denominators = torch.sqrt(cum_sq_from_back[:, :-1] + eps)

    numerators = e[:, :-1]
    ratio = torch.clamp(numerators / denominators, -1.0 + eps, 1.0 - eps)
    angles = torch.acos(ratio)

    e_d = e[:, -1]
    last_angle = angles[:, -1]
    adjusted_last_angle = torch.where(
        e_d < 0, 2 * math.pi - last_angle, last_angle)

    angles[:, -1] = adjusted_last_angle

    theta = angles[:, -1]
    psi1 = angles[:, -2]
    psi2 = angles[:, -3]

    return theta, psi1, psi2


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
