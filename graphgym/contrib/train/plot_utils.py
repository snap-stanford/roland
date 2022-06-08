import os

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from graphgym.config import cfg


def visualize_attention(dataset, att_path, fig_path):
    """Some visualizations to """
    # Read attention weights.
    num_pred = len(dataset) - cfg.transaction.horizon
    att_weights = list()
    for i in range(num_pred):
        att_weights.append(torch.load(att_path + f'/{i}.pt'))

    # Integrity check.
    for i in range(num_pred):
        alpha, g = att_weights[i], dataset[i]
        assert alpha.shape[0] == g.edge_index.shape[1]

    try:
        os.mkdir(fig_path)
    except FileExistsError:
        print(
            '\x1b[5;37;41m' + fig_path + ' exists, over-write' + '\x1b[0m')

    # Get edges with high attentions and low attentions.
    high_list, low_list, all_list = list(), list(), list()
    writer = SummaryWriter(fig_path)
    for i in range(num_pred):
        alpha, g = att_weights[i], dataset[i]
        writer.add_histogram('attention scores',
                             alpha.detach().cpu().numpy(), i)
        # NOTE: This threshold is subject to change.
        lower, upper = torch.quantile(alpha, 0.2), torch.quantile(alpha,
                                                                  0.8)
        # lower, upper = 0.05, 0.5

        low_att_edges = g.edge_feature[alpha <= lower, :]
        high_att_edges = g.edge_feature[upper <= alpha, :]
        # high_att_edges = g.edge_feature[torch.logical_and(alpha >=
        # upper, alpha < 1), :]

        low_list.append(low_att_edges)
        high_list.append(high_att_edges)
        all_list.append(g.edge_feature)

    writer.close()

    low_att_edges = torch.cat(low_list, dim=0).detach().cpu().numpy()
    high_att_edges = torch.cat(high_list, dim=0).detach().cpu().numpy()
    all_edges = torch.cat(all_list, dim=0).detach().cpu().numpy()

    all_att = torch.cat(att_weights, dim=0)
    # feature dim for int edge features (ef).
    # ef_bank: 0, 1
    # ef_country: 2, 3
    # ef_region: 4, 5
    # ef_skd: L1(6, 7), L2(8, 9) Order: (payer, payee), (payer, payee).
    # ef_skis: L1(10, 11), L2(12, 13)  Order: (payer, payee), (payer, payee).
    # # System: 14,
    # Currency: 15
    # edge_amount, 16
    # edge_time: 17

    upper_truncate = lambda x, p: x[x <= np.quantile(x, p)] if p < 1 else x

    # ================ Attention Score ================
    fig, ax = plt.subplots()
    # ax.hist(all_att.detach().cpu().numpy(), alpha=0.5, bins=40)
    val = all_att.detach().cpu().numpy()
    sns.kdeplot(x=val, ax=ax, alpha=0.5, fill=True)
    ax.set_xlabel('attention score')
    # ax.set_ylabel('number of transactions')
    ax.set_ylabel('probability density')
    fig.savefig(os.path.join(fig_path, 'att.png'), dpi=150, bbox_inches=None)

    # ================ Transaction Amount ================
    # Truncate out top 10% for nicer graphs.
    low_amt = upper_truncate(low_att_edges[:, 16], 0.9)
    high_amt = upper_truncate(high_att_edges[:, 16], 0.9)
    fig, ax = plt.subplots()
    ax.hist(low_amt, label=f'low attention edge ({len(low_amt)}) amount',
            alpha=0.3, bins=40)
    ax.hist(high_amt, label=f'high attention edge ({len(high_amt)}) amount',
            alpha=0.3, bins=40)
    # sns.kdeplot(x=low_amt, ax=ax, fill=True,
    #             label='low attention edge amount', alpha=0.5)
    # sns.kdeplot(x=high_amt, ax=ax, fill=True,
    #             label='high attention edge amount', alpha=0.5)
    ax.set_xlabel('(normalized) transaction amount')
    # ax.set_xlabel('transaction amount (EUR)')
    ax.set_ylabel('density')
    ax.legend()
    fig.savefig(os.path.join(fig_path, 'amount.png'), dpi=150,
                bbox_inches=None)

    # Plot Pairs (node attributes of edges)
    subsample = lambda x, s: x[np.random.choice(len(x), s), :]

    # NOTE: need to subsample to speed up KDE plot.
    size = int(1e4)
    high_att_edges = subsample(high_att_edges, size)
    low_att_edges = subsample(low_att_edges, size)

    # Format: (edge_feature_i, edge_feature_j, feature_name),
    # see comments above for index-feature correspondence in edge_feature.
    plot_pairs = [
        (0, 1, 'bank', False),
        # (2, 3, 'country', False),
        (4, 5, 'region', False),
        (6, 7, 'SkdL1', False),
        (8, 9, 'SkdL2', False),
        (10, 11, 'SkisL1', False),
        (12, 13, 'SkisL2', False)
    ]

    for (i, j, name, trunc) in plot_pairs:
        print('Plotting  ' + name)
        # xi: feature of source nodes.
        # xj: feature of destination nodes.
        low_xi, low_xj = low_att_edges[:, i], low_att_edges[:, j]
        high_xi, high_xj = high_att_edges[:, i], high_att_edges[:, j]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        sns.kdeplot(x=low_xi, y=low_xj, ax=axes[0], fill=True, alpha=0.5)
        axes[0].set_title('low attention edges')

        sns.kdeplot(x=high_xi, y=high_xj, ax=axes[1], fill=True, alpha=0.5)
        axes[1].set_title('high attention edges')
        for ax in axes:
            ax.set_xlabel("sender's " + name + " ID")
            ax.set_ylabel("receiver's " + name + " ID")
        fig.savefig(os.path.join(fig_path, f'{name}.png'), dpi=150,
                    bbox_inches=None)

    # Grid of histograms, only applicable to vars with few distinct values.
    plot_pairs = [
        (2, 3, 'country', False),
        (4, 5, 'region', False),
        (6, 7, 'SkdL1', False),
        (10, 11, 'SkisL1', False),
        (12, 13, 'SkisL2', False)
    ]
    attention_scores = all_att.detach().cpu().numpy()
    for (i, j, name, trunc) in plot_pairs:
        print('Plotting  ' + name)
        # Get unique values of the current variable.
        unique_i = np.unique(all_edges[:, i])
        unique_j = np.unique(all_edges[:, j])
        all_vals = set(unique_i).union(set(unique_j))
        all_vals = sorted(list(all_vals))  # in ascending order .
        num_vals = len(all_vals)

        # Plot distribution of attention scores associated with edges with
        # each combination of (payer_var, payee_var).
        fig, axes = plt.subplots(num_vals, num_vals,
                                 figsize=(num_vals * 2, num_vals * 2),
                                 sharex='all',
                                 sharey='all')
        fig.text(0.5, 0.04, f'Sender {name}', ha='center')
        fig.text(0.04, 0.5, f'Recipient {name}', va='center',
                 rotation='vertical')

        for ax_i, val_i in enumerate(all_vals):
            for ax_j, val_j in enumerate(all_vals):
                # Get edges with current combination of values.
                mask = np.logical_and(
                    all_edges[:, i] == val_i,
                    all_edges[:, j] == val_j)
                # cur_edges = all_edges[mask, :]
                # Plot the distribution of attention scores.
                cur_att = attention_scores[mask]
                ax = axes[ax_i, ax_j]
                ax.set_xlim(-0.1, 1.1)
                sns.kdeplot(cur_att, ax=ax, fill=True, alpha=0.5)

        fig.savefig(os.path.join(fig_path, f'{name}_grid.png'), dpi=150,
                    bbox_inches=None)


def visualize_dataset(datasets, fig_path):
    """Plots some aspects of datasets"""
    try:
        os.mkdir(fig_path)
    except FileExistsError:
        print('\x1b[5;37;41m' + fig_path + ' exists, over-write' + '\x1b[0m')

    num_snapshots = len(datasets[0])
    snapshot_to_plot = np.random.choice(num_snapshots, size=5, replace=False)

    fig, axes = plt.subplots(nrows=2, ncols=len(snapshot_to_plot),
                             figsize=(20, 8))

    for i, t in enumerate(np.sort(snapshot_to_plot)):
        # Plot the training set.
        g = datasets[0][int(t)]
        trans = lambda x: np.log(x.numpy())
        for j in [0, 1]:  # j=0 counts out degrees, j=1 counts in degrees.
            # Compare degree distribution of positive and negative edges.
            nv, n_deg = torch.unique(g.edge_label_index[j, g.edge_label == 0],
                                     return_counts=True)
            pv, p_deg = torch.unique(g.edge_label_index[j, g.edge_label == 1],
                                     return_counts=True)

            ax = axes[j, i]
            direction = 'out' if j == 0 else 'in'
            if j == 0:
                ax.set_title(f'snapshot {t}, '
                             f'\n|pos E| = {torch.sum(g.edge_label)}')
            ax.set_xlabel(f'log({direction} degree)')
            ax.set_ylabel(f'node count')
            ax.hist(trans(n_deg), label='negative degree', alpha=0.5, bins=40)
            ax.hist(trans(p_deg), label='positive degree', alpha=0.5, bins=40)
            ax.legend()

    fig.savefig(os.path.join(fig_path, 'pos_neg_degree_distribution.png'),
                dpi=150, bbox_inches='tight')
