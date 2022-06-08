"""
Loader for the CollegeMsg temporal network.

For more information: https://snap.stanford.edu/data/CollegeMsg.html

Mar. 31, 2021
"""
import os
from typing import List, Union

import deepsnap
import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph
from sklearn.preprocessing import MinMaxScaler

from graphgym.config import cfg
from graphgym.contrib.loader.loader_utils import make_graph_snapshot
from graphgym.register import register_loader
from graphgym.utils.stats import node_degree


def load_single_dataset(dataset_dir: str) -> Graph:
    # Load dataset using dask for fast parallel loading.
    df_trans = pd.read_csv(dataset_dir, sep=' ', header=None)
    df_trans.columns = ['SRC', 'DST', 'TIMESTAMP']
    assert not np.any(pd.isna(df_trans).values)
    df_trans.reset_index(drop=True, inplace=True)

    # Node IDs of this dataset start from 1.
    df_trans['SRC'] -= 1
    df_trans['DST'] -= 1

    print('num of edges:', len(df_trans))
    print('num of nodes:', np.max(df_trans[['SRC', 'DST']].values) + 1)

    time_scaler = MinMaxScaler((0, 2))
    df_trans['TimestampScaled'] = time_scaler.fit_transform(
        df_trans['TIMESTAMP'].values.reshape(-1, 1))

    edge_feature = torch.Tensor(
        df_trans[['TimestampScaled']].values).view(-1, 1)
    edge_index = torch.Tensor(
        df_trans[['SRC', 'DST']].values.transpose()).long()  # (2, E)
    num_nodes = torch.max(edge_index) + 1

    # node_feature = node_degree(edge_index, n=num_nodes, mode='both')
    # node_feature = node_feature.view(num_nodes, 1)
    # use node degree as feature leads to info-leak risk.
    node_feature = torch.ones(num_nodes, 1)

    print('feature_node_int_num: ', node_feature.max() + 1)

    edge_time = torch.FloatTensor(df_trans['TIMESTAMP'].values)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )

    return graph


def split_by_seconds(g_all, freq_sec: int):
    # Split the entire graph into snapshots.
    split_criterion = g_all.edge_time // freq_sec
    groups = torch.sort(torch.unique(split_criterion))[0]
    snapshot_list = list()
    for t in groups:
        period_members = (split_criterion == t)
        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )
        snapshot_list.append(g_incr)
    return snapshot_list


def load_generic(dataset_dir: str,
                 snapshot: bool = True,
                 snapshot_freq: str = None
                 ) -> Union[deepsnap.graph.Graph,
                            List[deepsnap.graph.Graph]]:
    g_all = load_single_dataset(dataset_dir)
    if not snapshot:
        return g_all
    if snapshot_freq.upper() not in ['D', 'W', 'M']:
        # format: '1200000s'
        # assume split by seconds (timestamp) as in EvolveGCN paper.
        freq = int(snapshot_freq.strip('s'))
        snapshot_list = split_by_seconds(g_all, freq)

    else:
        snapshot_list = make_graph_snapshot(g_all, snapshot_freq,
                                            is_hetero=False)

    num_nodes = g_all.edge_index.max() + 1

    for g_snapshot in snapshot_list:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    return snapshot_list


def load_generic_dataset(format, name, dataset_dir):
    if format == 'uci_message':
        graphs = load_generic(os.path.join(dataset_dir, name),
                              snapshot=cfg.transaction.snapshot,
                              snapshot_freq=cfg.transaction.snapshot_freq)
        if cfg.dataset.split_method == 'chronological_temporal':
            # return graphs
            filtered_graphs = list()
            for g in graphs:
                if g.num_edges >= 2:
                    filtered_graphs.append(g)
            return filtered_graphs
        else:
            # The default split (80-10-10) requires at least 10 edges each
            # snapshot.
            filtered_graphs = list()
            for g in graphs:
                if g.num_edges >= 10:
                    filtered_graphs.append(g)
            return filtered_graphs


register_loader('roland_uci_message', load_generic_dataset)
