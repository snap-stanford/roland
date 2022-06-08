"""
This script includes training/validating/testing procedures for rolling scheme.
"""
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.register import register_train
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.utils.stats import node_degree

import copy

# --------------------------------------------------------------------------- #
# Helper Functions.
# --------------------------------------------------------------------------- #


def report_rank_based_eval_hetero(eval_batch, model):
    """
    A modified version of report_rank_based_eval that computes MRR and recall@K
    for heterogeneous graphs.

    Args:
        eval_batch: a clone of training batch, must clone the training batch
            since we will mess up attributes of it.
        model: the trained model.
    """
    # %% Construct additional negative edges for MRR computation.
    for msg_type in eval_batch.edge_label.keys():
        s, r, d = msg_type
        _pos_mask = (eval_batch.edge_label[msg_type] == 1)
        edge_index = eval_batch.edge_label_index[msg_type][:, _pos_mask].to(
            'cpu')  # all positive edges.

        s_size = eval_batch.node_feature[s].shape[0]
        d_size = eval_batch.node_feature[d].shape[0]

        idx = (edge_index[0] * d_size + edge_index[1]).to('cpu')
        uni_src = torch.unique(edge_index[0])

        # 1 positive against approx. `multiplier` negative for each src.
        multiplier = cfg.experimental.rank_eval_multiplier

        src = uni_src.repeat_interleave(multiplier)
        dst = torch.tensor(
            np.random.choice(d_size, multiplier * len(uni_src), replace=True))

        perm = (src * d_size + dst)
        mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
        perm = perm[~mask]  # Filter out false negative edges.
        row = perm // d_size
        col = perm % d_size
        neg_edge_index = torch.stack([row, col], dim=0).long()

        new_edge_label_index = torch.cat((edge_index, neg_edge_index),
                                         dim=1).long()
        new_edge_label = torch.cat((torch.ones(edge_index.shape[1]),
                                    torch.zeros(neg_edge_index.shape[1])
                                    ), dim=0).long()

        # Construct evaluation samples.
        eval_batch.edge_label_index[msg_type] = new_edge_label_index
        eval_batch.edge_label[msg_type] = new_edge_label

    # %% Move batch to gpu.
    eval_batch.to(torch.device(cfg.device))
    for key in eval_batch.node_types:
        for layer in range(len(eval_batch.node_states[key])):
            if torch.is_tensor(eval_batch.node_states[key][layer]):
                eval_batch.node_states[key][layer] = \
                    eval_batch.node_states[key][layer].to(
                        torch.device(cfg.device))

    with torch.no_grad():
        pred, true = model(eval_batch)  # dictionaries.

    mrr = list()
    recall = {1: [], 3: [], 10: []}

    # %% Collapse (S,*,*)
    nt = eval_batch.node_types
    pred_score_by_sender = dict((s, []) for s in nt)
    edge_label_index_by_sender = dict((s, []) for s in nt)
    edge_label_by_sender = dict((s, []) for s in nt)

    for msg_type in eval_batch.edge_label.keys():
        # compute score for each message type.
        s, r, d = msg_type
        loss, pred_score = compute_loss(pred[msg_type], true[msg_type])
        edge_label_index = eval_batch.edge_label_index[msg_type]
        edge_label = eval_batch.edge_label[msg_type]

        pred_score_by_sender[s].append(pred_score)
        edge_label_index_by_sender[s].append(edge_label_index)
        edge_label_by_sender[s].append(edge_label)

    for s in eval_batch.node_types:
        p = torch.cat(pred_score_by_sender[s], dim=0)
        ei = torch.cat(edge_label_index_by_sender[s], dim=-1)
        y = torch.cat(edge_label_by_sender[s], dim=0)

        for sid in tqdm(range(eval_batch.num_nodes(s))):
            self_mask = (ei[0] == sid)
            if not torch.any(self_mask):
                continue  # This sender did not send anything.
            self_label = y[self_mask]
            self_pred_score = p[self_mask]

            neg_scores = self_pred_score[self_label == 0]
            best_pos_score = torch.max(self_pred_score[self_label == 1])

            if len(neg_scores) > 0:
                rank = torch.sum((best_pos_score <= neg_scores).float()) + 1
            else:
                rank = 1

            mrr.append(1 / float(rank))

            for k in [1, 3, 10]:
                recall[k].append(int(float(rank) <= k))

    mrr = np.mean(mrr)
    rck1 = np.mean(recall[1])
    rck3 = np.mean(recall[3])
    rck10 = np.mean(recall[10])

    print(f"MRR = {mrr})")
    print(f"avg Recall@1 = {rck1}, @3={rck3}, @10={rck10}")

    return mrr, rck1, rck3, rck10


# def visualize_attention(dataset, att_path, fig_path):
#     """Some visualizations to """
#     # Read attention weights.
#     num_pred = len(dataset) - cfg.transaction.horizon
#     att_weights = list()
#     for i in range(num_pred):
#         att_weights.append(torch.load(att_path + f'/{i}.pt'))
#
#     # Integrity check.
#     for i in range(num_pred):
#         alpha, g = att_weights[i], dataset[i]
#         assert alpha.shape[0] == g.edge_index.shape[1]
#
#     try:
#         os.mkdir(fig_path)
#     except FileExistsError:
#         print(
#             '\x1b[5;37;41m' + fig_path + ' exists, over-write' + '\x1b[0m')
#
#     # Get edges with high attentions and low attentions.
#     high_list, low_list, all_list = list(), list(), list()
#     writer = SummaryWriter(fig_path)
#     for i in range(num_pred):
#         alpha, g = att_weights[i], dataset[i]
#         writer.add_histogram('attention scores',
#                              alpha.detach().cpu().numpy(), i)
#         # NOTE: This threshold is subject to change.
#         lower, upper = torch.quantile(alpha, 0.2), torch.quantile(alpha,
#                                                                   0.8)
#         # lower, upper = 0.05, 0.5
#
#         low_att_edges = g.edge_feature[alpha <= lower, :]
#         high_att_edges = g.edge_feature[upper <= alpha, :]
#         # high_att_edges = g.edge_feature[torch.logical_and(alpha >=
#         # upper, alpha < 1), :]
#
#         low_list.append(low_att_edges)
#         high_list.append(high_att_edges)
#         all_list.append(g.edge_feature)
#
#     writer.close()
#
#     low_att_edges = torch.cat(low_list, dim=0).detach().cpu().numpy()
#     high_att_edges = torch.cat(high_list, dim=0).detach().cpu().numpy()
#     all_edges = torch.cat(all_list, dim=0).detach().cpu().numpy()
#
#     all_att = torch.cat(att_weights, dim=0)
#     # feature dim for int edge features (ef).
#     # ef_bank: 0, 1
#     # ef_country: 2, 3
#     # ef_region: 4, 5
#     # ef_skd: L1(6, 7), L2(8, 9) Order: (payer, payee), (payer, payee).
#     # ef_skis: L1(10, 11), L2(12, 13)  Order: (payer, payee), (payer, payee).
#     # # System: 14,
#     # Currency: 15
#     # edge_amount, 16
#     # edge_time: 17
#
#     upper_truncate = lambda x, p: x[x <= np.quantile(x, p)] if p < 1 else x
#
#     # ================ Attention Score ================
#     fig, ax = plt.subplots()
#     # ax.hist(all_att.detach().cpu().numpy(), alpha=0.5, bins=40)
#     val = all_att.detach().cpu().numpy()
#     sns.kdeplot(x=val, ax=ax, alpha=0.5, fill=True)
#     ax.set_xlabel('attention score')
#     # ax.set_ylabel('number of transactions')
#     ax.set_ylabel('probability density')
#     fig.savefig(os.path.join(fig_path, 'att.png'), dpi=150, bbox_inches=None)
#
#     # ================ Transaction Amount ================
#     # Truncate out top 10% for nicer graphs.
#     low_amt = upper_truncate(low_att_edges[:, 16], 0.9)
#     high_amt = upper_truncate(high_att_edges[:, 16], 0.9)
#     fig, ax = plt.subplots()
#     ax.hist(low_amt, label=f'low attention edge ({len(low_amt)}) amount',
#             alpha=0.3, bins=40)
#     ax.hist(high_amt, label=f'high attention edge ({len(high_amt)}) amount',
#             alpha=0.3, bins=40)
#     # sns.kdeplot(x=low_amt, ax=ax, fill=True,
#     #             label='low attention edge amount', alpha=0.5)
#     # sns.kdeplot(x=high_amt, ax=ax, fill=True,
#     #             label='high attention edge amount', alpha=0.5)
#     ax.set_xlabel('(normalized) transaction amount')
#     # ax.set_xlabel('transaction amount (EUR)')
#     ax.set_ylabel('density')
#     ax.legend()
#     fig.savefig(os.path.join(fig_path, 'amount.png'), dpi=150,
#                 bbox_inches=None)
#
#     # Plot Pairs (node attributes of edges)
#     subsample = lambda x, s: x[np.random.choice(len(x), s), :]
#
#     # NOTE: need to subsample to speed up KDE plot.
#     size = int(1e4)
#     high_att_edges = subsample(high_att_edges, size)
#     low_att_edges = subsample(low_att_edges, size)
#
#     # Format: (edge_feature_i, edge_feature_j, feature_name),
#     # see comments above for index-feature correspondence in edge_feature.
#     plot_pairs = [
#         (0, 1, 'bank', False),
#         # (2, 3, 'country', False),
#         (4, 5, 'region', False),
#         (6, 7, 'SkdL1', False),
#         (8, 9, 'SkdL2', False),
#         (10, 11, 'SkisL1', False),
#         (12, 13, 'SkisL2', False)
#     ]
#
#     for (i, j, name, trunc) in plot_pairs:
#         print('Plotting  ' + name)
#         # xi: feature of source nodes.
#         # xj: feature of destination nodes.
#         low_xi, low_xj = low_att_edges[:, i], low_att_edges[:, j]
#         high_xi, high_xj = high_att_edges[:, i], high_att_edges[:, j]
#
#         fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
#         sns.kdeplot(x=low_xi, y=low_xj, ax=axes[0], fill=True, alpha=0.5)
#         axes[0].set_title('low attention edges')
#
#         sns.kdeplot(x=high_xi, y=high_xj, ax=axes[1], fill=True, alpha=0.5)
#         axes[1].set_title('high attention edges')
#         for ax in axes:
#             ax.set_xlabel("sender's " + name + " ID")
#             ax.set_ylabel("receiver's " + name + " ID")
#         fig.savefig(os.path.join(fig_path, f'{name}.png'), dpi=150,
#                     bbox_inches=None)
#
#     # Grid of histograms, only applicable to vars with few distinct values.
#     plot_pairs = [
#         (2, 3, 'country', False),
#         (4, 5, 'region', False),
#         (6, 7, 'SkdL1', False),
#         (10, 11, 'SkisL1', False),
#         (12, 13, 'SkisL2', False)
#     ]
#     attention_scores = all_att.detach().cpu().numpy()
#     for (i, j, name, trunc) in plot_pairs:
#         print('Plotting  ' + name)
#         # Get unique values of the current variable.
#         unique_i = np.unique(all_edges[:, i])
#         unique_j = np.unique(all_edges[:, j])
#         all_vals = set(unique_i).union(set(unique_j))
#         all_vals = sorted(list(all_vals))  # in ascending order .
#         num_vals = len(all_vals)
#
#         # Plot distribution of attention scores associated with edges with
#         # each combination of (payer_var, payee_var).
#         fig, axes = plt.subplots(num_vals, num_vals,
#                                  figsize=(num_vals*2, num_vals*2),
#                                  sharex='all',
#                                  sharey='all')
#         fig.text(0.5, 0.04, f'Sender {name}', ha='center')
#         fig.text(0.04, 0.5, f'Recipient {name}', va='center',
#                  rotation='vertical')
#
#         for ax_i, val_i in enumerate(all_vals):
#             for ax_j, val_j in enumerate(all_vals):
#                 # Get edges with current combination of values.
#                 mask = np.logical_and(
#                     all_edges[:, i] == val_i,
#                     all_edges[:, j] == val_j)
#                 # cur_edges = all_edges[mask, :]
#                 # Plot the distribution of attention scores.
#                 cur_att = attention_scores[mask]
#                 ax = axes[ax_i, ax_j]
#                 ax.set_xlim(-0.1, 1.1)
#                 sns.kdeplot(cur_att, ax=ax, fill=True, alpha=0.5)
#
#         fig.savefig(os.path.join(fig_path, f'{name}_grid.png'), dpi=150,
#                     bbox_inches=None)
#
#
# def visualize_dataset(datasets, fig_path):
#     """Plots some aspects of datasets"""
#     try:
#         os.mkdir(fig_path)
#     except FileExistsError:
#         print('\x1b[5;37;41m' + fig_path + ' exists, over-write' + '\x1b[0m')
#
#     num_snapshots = len(datasets[0])
#     snapshot_to_plot = np.random.choice(num_snapshots, size=5, replace=False)
#
#     fig, axes = plt.subplots(nrows=2, ncols=len(snapshot_to_plot),
#                              figsize=(20, 8))
#
#     for i, t in enumerate(np.sort(snapshot_to_plot)):
#         # Plot the training set.
#         g = datasets[0][int(t)]
#         trans = lambda x: np.log(x.numpy())
#         for j in [0, 1]:  # j=0 counts out degrees, j=1 counts in degrees.
#             # Compare degree distribution of positive and negative edges.
#             nv, n_deg = torch.unique(g.edge_label_index[j, g.edge_label == 0],
#                                      return_counts=True)
#             pv, p_deg = torch.unique(g.edge_label_index[j, g.edge_label == 1],
#                                      return_counts=True)
#
#             ax = axes[j, i]
#             direction = 'out' if j == 0 else 'in'
#             if j == 0:
#                 ax.set_title(f'snapshot {t}, '
#                              f'\n|pos E| = {torch.sum(g.edge_label)}')
#             ax.set_xlabel(f'log({direction} degree)')
#             ax.set_ylabel(f'node count')
#             ax.hist(trans(n_deg), label='negative degree', alpha=0.5, bins=40)
#             ax.hist(trans(p_deg), label='positive degree', alpha=0.5, bins=40)
#             ax.legend()
#
#     fig.savefig(os.path.join(fig_path, 'pos_neg_degree_distribution.png'),
#                 dpi=150, bbox_inches='tight')
#
#
# def create_eval_label(eval_batch):
#     # Get positive edge indices.
#     edge_index = eval_batch.edge_label_index[:, eval_batch.edge_label == 1].to(
#         "cpu")
#     # idx = N * i + j
#     idx = (edge_index[0] * eval_batch.num_nodes + edge_index[1]).to("cpu")
#
#     # Generate negative edges, get senders of positive edges.
#     senders = torch.unique(edge_index[0]).detach().cpu()
#     # Consider these senders as users, sample negative edges for each user.
#     multiplier = 10  # (approximately) how many negative edges for each sender.
#     senders = senders.repeat_interleave(multiplier)
#     random_receivers = torch.tensor(np.random.choice(
#         eval_batch.num_nodes, len(senders), replace=True))
#
#     perm = (senders * eval_batch.num_nodes + random_receivers)
#     mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
#     perm = perm[~mask]  # Filter out false negative edges.
#     row = perm // eval_batch.num_nodes
#     col = perm % eval_batch.num_nodes
#     neg_edge_index = torch.stack([row, col], dim=0).long()
#
#     new_edge_label_index = torch.cat((edge_index, neg_edge_index), dim=1).long()
#     new_edge_label = torch.cat((torch.ones(edge_index.shape[1]),
#                                 torch.zeros(neg_edge_index.shape[1])
#                                 ), dim=0).long()
#
#     eval_batch.edge_label_index = new_edge_label_index.to(
#         torch.device(cfg.device))
#     eval_batch.edge_label = new_edge_label.to(torch.device(cfg.device))
#
#     eval_batch.senders = senders
#     return eval_batch


# def compute_rank_metrics(eval_batch, pred_score):
#     # pos_scores = pred_score[eval_batch.edge_label == 1]
#     # neg_scores = pred_score[eval_batch.edge_label == 0]
#
#     # Compute MRR across edges.
#     # pos_node_candidates = torch.unique(edge_index[0])
#     # neg_node_candidates = torch.unique(eval_batch.edge_label_index[0])
#     # node_candidates = torch.unique(eval_batch.edge_label_index[0])
#     mrr_list = list()
#     for node in tqdm(eval_batch.senders):
#         breakpoint()
#         self_mask = (eval_batch.edge_label_index[0] == node)
#         self_label = eval_batch.edge_label[self_mask]
#         self_edge_label_index = eval_batch.edge_label_index[:, self_mask]
#         self_pred_score = pred_score[self_mask]
#
#         neg_scores = self_pred_score[self_label == 0]
#         best_pos_score = torch.max(self_pred_score[self_label == 1])
#         if len(neg_scores) > 0:
#             # If best_pos_score is the highest, none of neg_scores should be
#             # higher than it.
#             rank = torch.sum(best_pos_score <= neg_scores) + 1
#         else:
#             rank = 1
#         mrr_list.append(1 / float(rank))
#
#     print(f"MRR = {np.mean(mrr_list)}")


# --------------------------------------------------------------------------- #
# Training Modules.
# --------------------------------------------------------------------------- #


def get_edge_label(dataset, current, horizon, mode):
    if mode == 'before':
        raise NotImplementedError
        edge_label = torch.cat([dataset[current + i].edge_label
                                for i in range(1, horizon + 1)], dim=0)
        edge_label_index = torch.cat([dataset[current + i].edge_label_index
                                      for i in range(1, horizon + 1)], dim=1)
    elif mode == 'at':
        edge_label = copy.deepcopy(dataset[current + horizon].edge_label)
        edge_label_index = copy.deepcopy(
            dataset[current + horizon].edge_label_index)
    return edge_label, edge_label_index


def get_keep_ratio(existing, new, mode: str = 'linear'):
    """
    Get the keep ratio for individual nodes to update node embeddings.
    Specifically:
       state[v,t] = state[v,t-1]*keep_ratio + new_feature[v,t]*(1-keep_ratio)

    Args:
        existing: a tensor of nodes' degrees in G[0], G[1], ..., G[t-1].
        new: a tensor of nodes' degrees in G[t].
        mode: how to compute the keep_ratio.

    Returns:
        A tensor with shape (num_nodes,) valued in [0, 1].
    """
    if mode == 'constant':
        # This scheme is equivalent to exponential decaying.
        ratio = torch.ones_like(existing)
        # node observed for the first time, keep_ratio = 0.
        ratio[torch.logical_and(existing == 0, new > 0)] = 0
        # take convex combination of old and new embeddings.
        # 1/2 can be changed to other values.
        ratio[torch.logical_and(existing > 0, new > 0)] = 1 / 2
        # inactive nodes have keep ratio 1, embeddings don't change.
    elif mode == 'linear':
        # The original method proposed by Jiaxuan.
        ratio = existing / (existing + new + 1e-6)
    # Following methods aim to shrink the weight of existing
    # degrees, help to ensure non-trivial embedding update when the graph
    # is large and history is long.
    elif mode == 'log':
        ratio = torch.log(existing + 1) / (
            torch.log(existing + 1) + new + 1e-6)
    elif mode == 'sqrt':
        ratio = torch.sqrt(existing) / (torch.sqrt(existing) + new + 1e-6)
    else:
        raise NotImplementedError(f'Mode {mode} is not supported.')
    return ratio


def update_batch(batch, batch_new, mode):
    if mode == 'replace':
        for key in batch_new.node_types:
            for i in range(len(batch_new.node_states[key])):
                batch_new.node_states[key][i] \
                    = batch.node_states[key][i].detach().cpu()
            batch_new.node_degree_existing[key] \
                = batch.node_degree_existing[key].detach().cpu()
        return batch_new
    elif mode == 'concat':
        raise NotImplementedError
        keys = ['edge_feature', 'edge_index', 'edge_time']
        for key in keys:
            dim = 1 if 'index' in key else 0
            batch[key] = torch.cat([batch[key], batch_new[key]], dim=dim)
        return batch


def train_epoch(logger, model, optimizer, scheduler, dataset, train=True,
                report_rank_based_metric=False):
    """A single epoch of training, validating or testing.
    """
    if train:
        model.train()
    else:
        model.eval()
    time_start = time.time()

    mrr_lst, rck1_lst, rck3_lst, rck10_lst = [], [], [], []

    rng = range(len(dataset) - cfg.transaction.horizon)
    # if train:
    #     # Only train on a smaller subset of periods.
    #     if cfg.experimental.restrict_training_set != -1:
    #         assert cfg.experimental.restrict_training_set > 1
    #         num_periods = int(cfg.experimental.restrict_training_set)
    #         # Option 1: take the first k.
    #         rng = list(range(num_periods))
    #         # Option 2: randomly pick k in chronological order.
    #         # rng = [0] + random.sample(rng, k=num_periods-1)
    #         # rng = sorted(list(set(rng)))

    # for i in range(len(dataset) - cfg.transaction.horizon):
    for i in rng:
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        # using full history
        if cfg.transaction.history == 'full':
            raise NotImplementedError
            # if i == 0:
            #     batch_history = dataset[i]
            # else:
            #     batch_history = update_batch(batch_history, dataset[i],
            #                                  mode='concat')
            # batch = batch_history.clone()
        # using rolling history
        elif cfg.transaction.history == 'rolling':
            if i == 0:
                batch = dataset[i].clone()
            else:
                # update raw features
                batch = update_batch(batch, dataset[i].clone(), mode='replace')
            # print(batch.node_degree_existing)
            batch.node_degree_new = dict()
            for t in batch.node_types:
                n = batch.node_feature[t].shape[0]
                batch.node_degree_new[t] = torch.zeros(n)

            for msg_type in batch.edge_label.keys():
                s, r, d = msg_type
                # only care about the in-degree (s).
                # TODO: design choice, change this.
                s_degree = node_degree(batch.edge_index[msg_type],
                                       n=batch.node_degree_existing[s].shape[0],
                                       mode='in')
                batch.node_degree_new[s] += s_degree

            batch.keep_ratio = dict()
            for t in batch.node_types:
                batch.keep_ratio[t] = get_keep_ratio(
                    existing=batch.node_degree_existing[t],
                    new=batch.node_degree_new[t],
                    mode=cfg.transaction.keep_ratio
                ).unsqueeze(-1)
                batch.node_degree_existing[t] += batch.node_degree_new[t]

            # temporary fix: use keep_ratio.
            # for node_type in batch.node_types:
            #     batch.keep_ratio[node_type] = 0.5 * torch.ones(
            #         batch.node_feature[node_type].shape[0]).reshape(-1, 1)
        else:
            raise ValueError(
                f'Unsupported training mode: {cfg.transaction.history}')

        # set edge labels
        edge_label, edge_label_index = get_edge_label(dataset, i,
                                                      cfg.transaction.horizon,
                                                      cfg.transaction.pred_mode)

        batch.edge_label = edge_label
        batch.edge_label_index = edge_label_index

        # Uncomment to use time encoding for time positional encoding.
        # pred_time = min(dataset[i+1].edge_time)
        # batch.edge_time_delta = pred_time - batch.edge_time
        eval_batch = batch.clone()

        if cfg.transaction.loss == 'meta':
            raise NotImplementedError
        elif cfg.transaction.loss == 'supervised':
            batch.to(torch.device(cfg.device))
            # move state to gpu
            for key in batch.node_types:
                for layer in range(len(batch.node_states[key])):
                    if torch.is_tensor(batch.node_states[key][layer]):
                        batch.node_states[key][layer] = batch.node_states[key][
                            layer].to(
                            torch.device(cfg.device))
            pred, true = model(batch)  # already hetero operation.
            # TODO: compute loss by type, need to update logger.
            pred = torch.cat(list(pred.values()), dim=0)
            true = torch.cat(list(true.values()), dim=0)
        else:
            raise ValueError(f'Invalid loss: {cfg.transaction.loss}')

        loss, pred_score = compute_loss(pred, true)
        if train:
            loss.backward()
            optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)

        if report_rank_based_metric:
            mrr, rck1, rck3, rck10 = report_rank_based_eval_hetero(eval_batch,
                                                                   model)
            mrr_lst.append(mrr)
            rck1_lst.append(rck1)
            rck3_lst.append(rck3)
            rck10_lst.append(rck10)

        # Report stats for the current period.
        # print(
        #     f'[Period {i}] edge_label@{batch.edge_label.size()};'
        #     f'edge_label_index@{batch.edge_label_index.size()}')
        auc = roc_auc_score(true.detach().cpu().numpy(),
                            pred_score.detach().cpu().numpy())
        print(f'[Period {i}] Loss: {loss.item():.3f}, auc: {auc}')
        print(f'[Period {i}] Time taken: {time.time() - time_start}')

        time_start = time.time()

    if train:
        scheduler.step()

    if report_rank_based_metric:
        print(f'[Test/Val] Average MRR over periods: {np.mean(mrr_lst)}')
        print(f'[Test/Val] Average RC1 over periods: {np.mean(rck1_lst)}')
        print(f'[Test/Val] Average RC3 over periods: {np.mean(rck3_lst)}')
        print(f'[Test/Val] Average RC10 over periods: {np.mean(rck10_lst)}')


def train_example(loggers, loaders, model, optimizer, scheduler, datasets,
                  **kwargs):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], model, optimizer, scheduler,
                    datasets[0], train=True,
                    report_rank_based_metric=False)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                train_epoch(loggers[i], model, optimizer, scheduler,
                            datasets[i], train=False,
                            report_rank_based_metric=True)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    # if cfg.experimental.visualize_gnn_layer:
    #     # Analysis the attention on the validation set.
    #     # Save attention weights to disk for later visualization.
    #     cur_time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
    #     for i in range(cfg.gnn.layers_mp):
    #         getattr(model.mp, f'layer{i}').layer.model.save_att_weights(
    #             f'./{cur_time}_att_weights_layer{i}')
    #
    #     # Run an extra epoch to get attention.
    #     # TODO: choose the dataset you want!
    #     train_epoch(loggers[1], model, optimizer, scheduler,
    #                 datasets[1], train=False,
    #                 report_rank_based_metric=False)

    # for logger in loggers:
    #     logger.close()
    # if cfg.train.ckpt_clean:
    #     clean_ckpt()
    #
    # logging.info('Task done, results saved in {}'.format(cfg.out_dir))
    #


register_train('new_hetero', train_example)
