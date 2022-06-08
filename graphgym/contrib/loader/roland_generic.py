"""
A generic loader for the roland project, modify this template to build
loaders for other financial transaction datasets.
Mar. 22, 2021
"""
import os
from datetime import datetime
from typing import List, Union

import dask.dataframe as dd
import deepsnap
import numpy as np
import pandas as pd
import torch
from dask_ml.preprocessing import OrdinalEncoder
from deepsnap.graph import Graph
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder as SkOrdinalEncoder

from graphgym.config import cfg
from graphgym.register import register_loader

# =============================================================================
# Configure and instantiate the loader here.
# =============================================================================
# TODO: set attributes here.
# Required for all graphs.
SRC_NODE: str = 'Payer'
DST_NODE: str = 'Payee'
TIMESTAMP: str = 'Timestamp'
AMOUNT: str = 'AmountEUR'

# Categorical columns are SRC_NODE+var and DST_NODE+var.
# Note that '' corresponds to columns SRC_NODE and DST_NODE.
NODE_CATE_VARS: List[str] = ['', 'Bank', 'Country', 'Region', 'Skd', 'SkdL1',
                             'SkdL2', 'Skis', 'SkisL1', 'SkisL2']
EDGE_CATE_VARS: List[str] = ['# System', 'Currency']

EDGE_FEATURE_COLS: List[str] = ['PayerBank', 'PayeeBank',
                                'PayerCountry', 'PayeeCountry',
                                'PayerRegion', 'PayeeRegion',
                                'PayerSkdL1', 'PayeeSkdL1',
                                'PayerSkdL2', 'PayeeSkdL2',
                                'PayerSkisL1', 'PayeeSkisL1',
                                'PayerSkisL2', 'PayeeSkisL2',
                                '# System', 'Currency',
                                AMOUNT, 'TimestampScaled']

# Required for heterogeneous graphs only.
# Node and edge features used to define node and edge type in hete GNN.
NODE_TYPE_DEFN: List[str] = ['Country']
EDGE_TYPE_DEFN: List[str] = ['# System']

# Required for graphs with node features only.
NODE_FEATURE_LIST: List[str] = ['Bank', 'Country', 'Region', 'SkdL1', 'SkisL1']


# =============================================================================
# Helper Functions.
# =============================================================================

# Some helper functions, inputs should be a timestamp integer.
# See http://strftime.org for references.
def dayofyear(t: int) -> int:
    # '%j': Day of the year as a zero-padded decimal number.
    return int(datetime.fromtimestamp(t).strftime('%j'))


def weekofyear(t: int) -> int:
    # '%W' Week number of the year (Monday as the first day of the week)
    # as a decimal number.
    # All days in a new year preceding the first Monday are considered to be
    # in week 0.
    return int(datetime.fromtimestamp(t).strftime('%W'))


def monthofyear(t: int) -> int:
    # Get the month of year.
    return int(datetime.fromtimestamp(t).month)


def quarterofyear(t: int) -> int:
    m = datetime.fromtimestamp(t).month
    return int((m - 1) // 3 + 1)


def get_node_feature(df: pd.DataFrame,
                     required_features: List[str]) -> pd.DataFrame:
    """Extract node features from a transaction dataset.
    """
    char_lst = list()
    for party in [SRC_NODE, DST_NODE]:
        # ['Payer', 'PayerBank', 'PayerCountry', ...]
        cols = [party] + [party + var for var in required_features]
        relevant = df[cols].copy()
        # ['Company', 'Bank', 'Country', ...]
        relevant.columns = ['Company'] + required_features
        char_lst.append(relevant)
    df_char = pd.concat(char_lst, axis=0)

    df_char = df_char.groupby('Company').first()
    df_char = df_char[required_features]
    return df_char


def construct_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs additional features of the transaction dataset.
    """
    # TODO: construct customized transaction feature here.
    #  e.g., df['LargeTransaction'] = (df['Amount'] >= 1000)
    return df


def load_single_dataset(dataset_dir: str, is_hetero: bool = True,
                        type_info_loc: str = 'append',
                        include_node_features: bool = False
                        ) -> Graph:
    """
    Loads a single graph object from tsv file.

    Args:
        dataset_dir: the path of tsv file to be loaded.
        is_hetero: whether to load heterogeneous graph.
        type_info_loc: 'append' or 'graph_attribute'.
        include_node_features: when set to True, load node features into
            graph.node_feature, otherwise, the loader assumes node features
            are subsumed in edge_features, and set graph.node_feature to all 1.

    Returns:
        graph: a (homogenous) deepsnap graph object.
    """
    # Load dataset using dask for fast parallel loading.
    df_trans = dd.read_csv(dataset_dir, sep='\t', low_memory=False)
    df_trans = df_trans.fillna('missing')
    df_trans = df_trans.compute()

    df_trans = construct_additional_features(df_trans)
    df_trans.reset_index(drop=True, inplace=True)

    # a unique values of node-level categorical variables.
    node_cat_uniques = dict()  # Dict[str, np.ndarray]
    for var in NODE_CATE_VARS:
        relevant = df_trans[[SRC_NODE + var, DST_NODE + var]]
        unique_var = pd.unique(relevant.to_numpy().ravel())
        node_cat_uniques[var] = np.sort(unique_var)

    # %% Convert to pandas categorical variables data type for fast encoding.
    for var in NODE_CATE_VARS:
        unique_val = np.sort(node_cat_uniques[var])
        cate_type = pd.api.types.CategoricalDtype(categories=unique_val,
                                                  ordered=True)
        df_trans[SRC_NODE + var] = df_trans[SRC_NODE + var].astype(cate_type)
        df_trans[DST_NODE + var] = df_trans[DST_NODE + var].astype(cate_type)

    # Another 2 edge level categorical variables.
    for var in EDGE_CATE_VARS:
        unique_var = np.sort(pd.unique(df_trans[[var]].to_numpy().ravel()))
        cate_type = pd.api.types.CategoricalDtype(categories=unique_var,
                                                  ordered=True)
        df_trans[var] = df_trans[var].astype(cate_type)

    # %% Encoding categorical variables.
    enc = OrdinalEncoder()
    df_encoded = enc.fit_transform(df_trans)
    df_encoded.reset_index(drop=True, inplace=True)

    # Scaling transaction amount.
    scaler = MinMaxScaler((0, 2))
    df_encoded[AMOUNT] = scaler.fit_transform(
        df_encoded[AMOUNT].values.reshape(-1, 1))

    time_scaler = MinMaxScaler((0, 2))
    df_encoded['TimestampScaled'] = time_scaler.fit_transform(
        df_encoded[TIMESTAMP].values.reshape(-1, 1))

    # %% Prepare for output.
    edge_feature = torch.Tensor(
        df_encoded[EDGE_FEATURE_COLS].values)  # (E, edge_dim)
    edge_index = torch.Tensor(
        df_encoded[[SRC_NODE, DST_NODE]].values.transpose()).long()  # (2, E)
    num_nodes = torch.max(edge_index) + 1
    assert num_nodes == len(node_cat_uniques[''])

    if include_node_features:
        df_char = get_node_feature(df_encoded, NODE_FEATURE_LIST)
        node_feature = torch.Tensor(df_char.astype(float).values)
        # print([int(torch.max(node_feature[:, i])) + 1 for i in range(len(NODE_FEATURE_LIST))])
    else:
        node_feature = torch.ones(num_nodes, 1).float()

    edge_time = torch.FloatTensor(df_encoded[TIMESTAMP].values)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )

    if is_hetero:
        char_lst = list()
        for party in [SRC_NODE, DST_NODE]:
            cols = [party] + [party + var for var in NODE_TYPE_DEFN]
            relevant = df_encoded[cols].copy()
            relevant.columns = ['Company'] + NODE_TYPE_DEFN
            char_lst.append(relevant)
        df_char = pd.concat(char_lst, axis=0)

        df_char = df_char.groupby('Company').first()
        df_char = df_char[NODE_TYPE_DEFN].astype(str)

        # Construct node type signatures.
        df_char['NodeType'] = df_char[NODE_TYPE_DEFN[0]].astype(str)
        for var in NODE_TYPE_DEFN[1:]:
            df_char['NodeType'] += ('--' + df_char[var].astype(str))

        node_type_enc = SkOrdinalEncoder()
        node_type_int = node_type_enc.fit_transform(
            df_char['NodeType'].values.reshape(-1, 1))
        node_type_int = torch.FloatTensor(node_type_int)
        assert len(node_type_int) == num_nodes

        # Construct edge type signatures.
        df_trans['EdgeType'] = df_trans[EDGE_TYPE_DEFN[0]].astype(str)
        for var in EDGE_TYPE_DEFN[1:]:
            df_trans['EdgeType'] += ('--' + df_trans[var].astype(str))

        edge_type_enc = SkOrdinalEncoder()
        edge_type_int = edge_type_enc.fit_transform(
            df_trans['EdgeType'].values.reshape(-1, 1))
        edge_type_int = torch.FloatTensor(edge_type_int)

        if type_info_loc == 'append':
            graph.edge_feature = torch.cat((graph.edge_feature, edge_type_int),
                                           dim=1)
            graph.node_feature = torch.cat((graph.node_feature, node_type_int),
                                           dim=1)
        elif type_info_loc == 'graph_attribute':
            graph.node_type = node_type_int.reshape(-1, )
            graph.edge_type = edge_type_int.reshape(-1, )
        else:
            raise ValueError(f'Unsupported type info loc: {type_info_loc}')

        graph.list_n_type = node_type_int.unique().long()
        graph.list_e_type = edge_type_int.unique().long()

    return graph


def make_graph_snapshot(g_all: Graph,
                        snapshot_freq: str,
                        is_hetero: bool = True) -> list:
    """
    Constructs a list of graph snapshots (Graph or HeteroGraph) based
        on g_all and snapshot_freq.

    Args:
        g_all: the entire homogenous graph.
        snapshot_freq: snapshot frequency.
        is_hetero: if make heterogeneous graphs.
    """
    t = g_all.edge_time.numpy().astype(np.int64)
    snapshot_freq = snapshot_freq.upper()

    period_split = pd.DataFrame(
        {'Timestamp': t,
         'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(g_all.edge_time)))

    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m'  # month of year.
                }

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)

    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
    # e.g., dictionary w/ key = (2021, 3) and val = array(edges).

    periods = sorted(list(period2id.keys()))
    snapshot_list = list()
    for p in periods:
        # unique IDs of edges in this period.
        period_members = period2id[p]
        assert np.all(period_members == np.unique(period_members))

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed,
            list_n_type=g_all.list_n_type if is_hetero else None,
            list_e_type=g_all.list_e_type if is_hetero else None,
        )
        if is_hetero and hasattr(g_all, 'node_type'):
            g_incr.node_type = g_all.node_type
            g_incr.edge_type = g_all.edge_type[period_members]
        snapshot_list.append(g_incr)
    return snapshot_list


def load_generic(dataset_dir: str,
                 snapshot: bool = True,
                 snapshot_freq: str = None,
                 is_hetero: bool = True,
                 type_info_loc: str = 'append',
                 include_node_features: bool = False
                 ) -> Union[deepsnap.graph.Graph,
                            List[deepsnap.graph.Graph]]:
    g_all = load_single_dataset(dataset_dir, is_hetero=is_hetero,
                                type_info_loc=type_info_loc,
                                include_node_features=include_node_features)
    if not snapshot:
        return g_all
    else:
        snapshot_list = make_graph_snapshot(g_all, snapshot_freq, is_hetero)
        num_nodes = g_all.edge_index.max() + 1

        for g_snapshot in snapshot_list:
            g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_degree_existing = torch.zeros(num_nodes)

        return snapshot_list


def load_generic_dataset(format, name, dataset_dir):
    # TODO: change the format name.
    if format == 'generic':
        dataset_dir = os.path.join(dataset_dir, name)
        graphs = load_generic(dataset_dir,
                              snapshot=cfg.transaction.snapshot,
                              snapshot_freq=cfg.transaction.snapshot_freq,
                              is_hetero=cfg.dataset.is_hetero,
                              type_info_loc=cfg.dataset.type_info_loc,
                              include_node_features=cfg.dataset.include_node_features)
        return graphs


# TODO: change the loader name.
register_loader('roland_generic', load_generic_dataset)

if __name__ == '__main__':
    # Example usage.
    # TODO: change the directory to your sample dataset.
    dataset_dir = '/lfs/hyperturing2/0/tianyudu/bsi/all/data/transactions_svt_2008.tsv'
    # a list of homogenous monthly transaction graphs.
    graphs = load_generic(dataset_dir,
                          snapshot=True,
                          snapshot_freq='M',
                          is_hetero=False,
                          type_info_loc='graph_attribute')

    # a list of heterogeneous weekly transaction graphs.
    hete_graphs = load_generic(dataset_dir,
                               snapshot=True,
                               snapshot_freq='W',
                               is_hetero=True,
                               type_info_loc='graph_attribute')
