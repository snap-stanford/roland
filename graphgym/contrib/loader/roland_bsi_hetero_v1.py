"""
This script contains method to load bsi dataset, load the entire dataset
as a single graph or a list of graph snapshots.
Additionally, this loader supports constructing heterogenous graphs.
Mar. 1, 2021
"""
import os
from datetime import datetime
from typing import List, Union, Optional, Dict, Tuple

import dask.dataframe as dd
import deepsnap
import numpy as np
import pandas as pd
import torch
from dask_ml.preprocessing import OrdinalEncoder
from deepsnap.graph import Graph
from deepsnap.hetero_graph import HeteroGraph
from sklearn.preprocessing import MinMaxScaler

from graphgym.config import cfg
from graphgym.register import register_loader


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


# =============================================================================
# Core Functions.
# =============================================================================
def construct_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs additional features of the transaction dataset.
    """
    for p in ('Payer', 'Payee'):
        # %% Location of companies.
        mask = (df[p + 'Country'] != 'SI')
        out_of_country = np.empty(len(df), dtype=object)
        out_of_country[mask] = 'OutOfCountry'
        out_of_country[~mask] = 'InCountry'
        df[p + 'OutOfCountry'] = out_of_country

    # %% Cross country transactions.
    # mask = (df['PayerCountry'] != df['PayeeCountry'])
    # missing_mask = np.logical_or(df['PayerCountry'] == 'missing',
    #                              df['PayeeCountry'] == 'missing')
    # cross_country = np.empty(len(df), dtype=object)
    # cross_country[mask] = 'CrossCountry'
    # cross_country[~mask] = 'WithinCountry'
    # cross_country[missing_mask] = 'Missing'
    # df['CrossCountry'] = cross_country

    # %% Cross region transactions.
    # mask = (df['PayerRegion'] != df['PayeeRegion'])
    # missing_mask = np.logical_or(df['PayerRegion'] == 'missing',
    #                              df['PayeeRegion'] == 'missing')
    # cross_region = np.empty(len(df), dtype=object)
    # cross_region[mask] = 'CrossRegion'
    # cross_region[~mask] = 'WithinRegion'
    # cross_region[missing_mask] = 'Missing'
    # df['CrossRegion'] = cross_region

    # %% Large amount transaction.
    # Need to tune this variable to make categories balanced.
    amount_level = np.empty(len(df), dtype=object)
    # ternary split.
    mask_small = df['AmountEUR'] < 500
    mask_medium = np.logical_and(df['AmountEUR'] >= 500,
                                 df['AmountEUR'] < 1000)
    mask_large = df['AmountEUR'] >= 1000
    amount_level[mask_small] = '$<500'
    amount_level[mask_medium] = '500<=$<1k'
    amount_level[mask_large] = '$>=1k'

    # binary split.
    # mask_small = df['AmountEUR'] < 500
    # mask_large = df['AmountEUR'] >= 500
    # amount_level[mask_small] = '$<500'
    # amount_level[mask_large] = '$>=500'

    df['AmountLevel'] = amount_level

    return df


def load_single_dataset(dataset_dir: str, is_hetero: bool = True,
                        node_type_defn: List[str] = ['OutOfCountry'],
                        edge_type_defn: List[str] = ['AmountLevel']
                        ) -> (Graph,
                              Dict[str, np.ndarray],
                              Dict[str, np.ndarray],
                              Dict[Tuple[str], np.ndarray]):
    """
    Loads a single graph object from tsv file.

    Args:
        dataset_dir: the path of tsv file to be loaded.

        is_hetero: whether to load heterogeneous graph.

        node_type_defn: a list of columns of the dataset used to define
            node types. Only necessary if is_hetero == True.

        edge_type_defn: a list of columns of the dataset used to define
            edge types. Only necessary if is_hetero == True.

    NOTE: please note the difference between edge type string and message
        type tuple.

    Returns:
        graph: a (homogenous) deepsnap graph object.

        node_type2id: a dictionary maps node type string to an array
            of integer indices of nodes belong to this type.

        edge_type2id: a dictionary maps edge type string to integer indices
            of edges belonging to this type.

        message_type2id: a dictionary maps (s, r, d) message type tuples
            to integer indices of edges belong to this type.
            Where (s, r, d) is a tuple of str: (NodeType, EdgeType, NodeType).

        For is_hetero=False, {node, edge}_type2id are two empty dictionaries.
    """
    # %% Load dataset from disk using dask parallel.
    df_trans = dd.read_csv(dataset_dir, sep='\t',
                           dtype={'# System': str,
                                  'Payer': str, 'PayerBank': str,
                                  'PayerCountry': str, 'PayerRegion': str,
                                  'PayerSkd': str, 'PayerSkdL1': str,
                                  'PayerSkdL2': str, 'PayerSkis': str,
                                  'PayerSkisL1': str, 'PayerSkisL2': str,
                                  'Payee': str, 'PayeeBank': str,
                                  'PayeeCountry': str, 'PayeeRegion': str,
                                  'PayeeSkd': str, 'PayeeSkdL1': str,
                                  'PayeeSkdL2': str, 'PayeeSkis': str,
                                  'PayeeSkisL1': str, 'PayeeSkisL2': str,
                                  'Currency': str, 'Amount': np.float32,
                                  'AmountEUR': np.float32,
                                  'Year': np.int32, 'Month': np.int32,
                                  'DayOfMonth': np.int32,
                                  'DayOfWeek': np.int32,
                                  'Timestamp': np.int64},
                           low_memory=False)

    # TODO: any better ways to handle missing observations?
    #  There already exists an 'unknown' type in the original dataset.
    df_trans = df_trans.fillna('missing')
    df_trans = df_trans.compute()  # approx 7 min.
    df_trans = construct_additional_features(df_trans)
    df_trans.reset_index(drop=True, inplace=True)
    # %% Node level categorical features.
    # Categorical columns are 'Payer'+var and 'Payee'+var.
    # Note that '' corresponds to columns 'Payer' and 'Payee'.
    node_cate_vars = ['', 'Bank', 'Country', 'Region', 'Skd', 'SkdL1', 'SkdL2',
                      'Skis', 'SkisL1', 'SkisL2']

    # a unique values of node-level categorical variables.
    node_cat_uniques = dict()  # Dict[str, np.ndarray]
    for var in node_cate_vars:
        relevant = df_trans[['Payer' + var, 'Payee' + var]]
        unique_var = pd.unique(relevant.to_numpy().ravel())
        node_cat_uniques[var] = np.sort(unique_var)

    # %% Convert to pandas categorical variables data type for fast encoding.
    for var in node_cate_vars:
        unique_val = np.sort(node_cat_uniques[var])
        cate_type = pd.api.types.CategoricalDtype(categories=unique_val,
                                                  ordered=True)
        df_trans['Payer' + var] = df_trans['Payer' + var].astype(cate_type)
        df_trans['Payee' + var] = df_trans['Payee' + var].astype(cate_type)

    # Another 2 edge level categorical variables.
    for var in ['# System', 'Currency']:
        unique_var = np.sort(pd.unique(df_trans[[var]].to_numpy().ravel()))
        cate_type = pd.api.types.CategoricalDtype(categories=unique_var,
                                                  ordered=True)
        df_trans[var] = df_trans[var].astype(cate_type)

    # %% Encoding categorical variables.
    enc = OrdinalEncoder()
    df_encoded = enc.fit_transform(df_trans)
    df_encoded.reset_index(drop=True, inplace=True)

    # %% Construct the homogenous graph.
    # (1) Load as an ordinary graph.
    # columns to be added as edge features.
    edge_feature_cols = ['PayerBank', 'PayeeBank',
                         'PayerCountry', 'PayeeCountry',
                         'PayerRegion', 'PayeeRegion',
                         'PayerSkdL1', 'PayeeSkdL1',
                         'PayerSkdL2', 'PayeeSkdL2',
                         'PayerSkisL1', 'PayeeSkisL1',
                         'PayerSkisL2', 'PayeeSkisL2',
                         '# System', 'Currency',
                         'AmountEUR', 'TimestampScaled']

    # Scaling transaction amount.
    scaler = MinMaxScaler((0, 2))
    df_encoded['AmountEUR'] = scaler.fit_transform(
        df_encoded['AmountEUR'].values.reshape(-1, 1))

    time_scaler = MinMaxScaler((0, 2))
    df_encoded['TimestampScaled'] = time_scaler.fit_transform(
        df_encoded['Timestamp'].values.reshape(-1, 1))

    # %% Prepare for output.
    edge_feature = torch.Tensor(
        df_encoded[edge_feature_cols].values)  # (E, edge_dim)
    edge_index = torch.Tensor(
        df_encoded[['Payer', 'Payee']].values.transpose()).long()  # (2, E)
    num_nodes = torch.max(edge_index) + 1
    assert num_nodes == len(node_cat_uniques[''])

    # TODO: # use dummy features or df_char.
    node_feature = torch.ones(num_nodes, 1).float()
    edge_time = torch.FloatTensor(df_encoded['Timestamp'].values)

    # The homogenous graph.
    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    print([torch.max(graph.edge_feature[:, i]).long().item() + 1 for i in range(16)])

    if not is_hetero:
        # when is_hetero = False, behave the same as roland_bsi.
        return graph, dict(), dict(), dict()

    # REMARK: different payer characteristic might be inferred from
    # payer role and payee role. Here we ignore this fact.
    # Construct company level characteristics.
    char_lst = list()
    for party in ['Payer', 'Payee']:
        cols = [party] + [party + var for var in node_type_defn]
        relevant = df_encoded[cols].copy()
        relevant.columns = ['Company'] + node_type_defn
        char_lst.append(relevant)
    # a data frame of company characteristics.
    df_char = pd.concat(char_lst, axis=0)

    # NOTE: assume companies' characteristics are stable, use the first
    # transaction of each payer/payee to identify characteristics.
    df_char = df_char.groupby('Company').first()
    df_char = df_char[node_type_defn].astype(str)

    # Construct node type signatures.
    df_char['NodeType'] = df_char[node_type_defn[0]].astype(str)
    for var in node_type_defn[1:]:
        df_char['NodeType'] += ('--' + df_char[var].astype(str))

    g = df_char.groupby('NodeType')
    node_type2id = g.indices

    for p in ('Payer', 'Payee'):
        p_type = df_encoded.join(df_char[['NodeType']], on=[p])['NodeType']
        df_trans[p + 'Type'] = p_type

    df_trans['EdgeType'] = df_trans[edge_type_defn[0]].astype(str)
    for var in edge_type_defn[1:]:
        df_trans['EdgeType'] += ('--' + df_trans[var].astype(str))

    g = df_trans.groupby('EdgeType')
    edge_type2id = g.indices  # key: str, value: array[int]

    # %% Construct message type maps.
    g = df_trans.groupby(['PayerType', 'EdgeType', 'PayeeType'])
    message_type2id = g.indices  # key: Tuple[str], values: array[int].

    return graph, node_type2id, edge_type2id, message_type2id


def make_hetero_graph(homo_graph, node_type2id, message_type2id):
    """
    Takes a homogenous graph and node/edge type map, build the deepsnap
    HeteroGraph object.
    """
    # %% Heterogeneous nodes.
    hete_node_feature = dict()
    for node_type, node_indices in node_type2id.items():
        assert type(node_type) is str and type(node_indices) is np.ndarray
        hete_node_feature[node_type] = homo_graph.node_feature[node_indices,
                                       :].clone()
    # %% Heterogeneous edges.
    hete_edge_feature, hete_edge_index, hete_edge_time = dict(), dict(), dict()
    for msg_type, idx in message_type2id.items():
        s, r, d = msg_type

        hete_edge_feature[msg_type] = homo_graph.edge_feature[idx, :].clone()
        hete_edge_index[msg_type] = homo_graph.edge_index[:, idx].clone()
        src = hete_edge_index[msg_type][0, :]
        dst = hete_edge_index[msg_type][1, :]

        # src[k]: index of the src node of the k-th edge AMONG ALL NODES.
        # src_idx_within_type[k]: index of src node of hte k-th edge AMONG
        # NOTES WITH CURRENT SENDER TYPE S.
        src_idx_within_type = torch.searchsorted(
            torch.LongTensor(node_type2id[s]), src)
        dst_idx_within_type = torch.searchsorted(
            torch.LongTensor(node_type2id[d]), dst)

        hete_edge_index[msg_type] = torch.stack(
            (src_idx_within_type, dst_idx_within_type), dim=0)

        hete_edge_time[msg_type] = homo_graph.edge_time[idx].clone()

    hete = HeteroGraph(node_feature=hete_node_feature,
                       edge_feature=hete_edge_feature,
                       edge_index=hete_edge_index,
                       edge_time=hete_edge_time,
                       directed=homo_graph.directed)
    # hete._node_related_key = 'node_feature'  # smart enough to get this.
    return hete


def make_graph_snapshot(g_all: Graph,
                        snapshot_freq: str,
                        node_type2id: Optional[dict] = None,
                        message_type2id: Optional[dict] = None,
                        is_hetero: bool = True) -> list:
    """
    Constructs a list of graph snapshots (Graph or HeteroGraph) based
        on g_all and snapshot_freq.

    Args:
        g_all: the entire homogenous graph.
        snapshot_freq: snapshot frequency.
        node_type2id: a dictionary maps a node type string to a numpy array
            of IDs of nodes belonging to this type.
        message_type2id: a dictionary maps an edge type string to a numpy array
            of IDs of edges belonging to this type.
        is_hetero: if make heterogeneous graphs.
    """
    if is_hetero:
        if node_type2id is None or message_type2id is None:
            raise ValueError('HeteroGraph node/edge map required.')

    # t = pd.to_datetime(g_all.edge_time.numpy(), unit='s')
    t = g_all.edge_time.numpy().astype(np.int64)
    snapshot_freq = snapshot_freq.upper()

    period_split = pd.DataFrame({'TransactionTime': t},
                                index=range(len(g_all.edge_time)))
    # e.g., dict(month, array(edges in this month)), split based on freq.
    if snapshot_freq == 'D':
        period_split['SplitFlag'] = period_split['TransactionTime'].apply(
            dayofyear)
    elif snapshot_freq == 'W':
        period_split['SplitFlag'] = period_split['TransactionTime'].apply(
            weekofyear)
    elif snapshot_freq == 'M':
        period_split['SplitFlag'] = period_split['TransactionTime'].apply(
            monthofyear)
    elif snapshot_freq == 'Q':
        period_split['SplitFlag'] = period_split['TransactionTime'].apply(
            quarterofyear)
    else:
        raise NotImplementedError

    period2id = period_split.groupby('SplitFlag').indices
    assert np.issubdtype(type(list(period2id.keys())[0]), np.integer)
    assert type(list(period2id.values())[0]) is np.ndarray

    periods = sorted(list(period2id.keys()))
    snapshot_list = list()
    for p in periods:
        # unique IDs of edges in this period.
        period_members = period2id[p]
        assert np.all(period_members == np.unique(period_members))
        if is_hetero:
            # edge map for this graph snapshot.
            snapshot_msg_type2id = dict()
            for msg_type, type_members in message_type2id.items():
                assert type(msg_type) is tuple
                assert type(type_members) is np.ndarray
                # edge IDs (in the entire graph) belongs to this period
                # and this type.
                members = np.intersect1d(period_members, type_members)
                snapshot_msg_type2id[msg_type] = members
            g_incr = make_hetero_graph(homo_graph=g_all,
                                       # nodes are carried forward, only
                                       # modify edges.
                                       node_type2id=node_type2id,
                                       message_type2id=snapshot_msg_type2id
                                       )
        else:
            # homogenous graph.
            g_incr = Graph(
                node_feature=g_all.node_feature,
                edge_feature=g_all.edge_feature[period_members, :],
                edge_index=g_all.edge_index[:, period_members],
                edge_time=g_all.edge_time[period_members],
                directed=g_all.directed
            )

        snapshot_list.append(g_incr)
    return snapshot_list


def load_bsi_hetero(dataset_dir: str,
                    snapshot: bool = True,
                    snapshot_freq: str = None,
                    is_hetero: bool = True
                    ) -> Union[deepsnap.graph.Graph,
                               List[deepsnap.graph.Graph],
                               deepsnap.hetero_graph.HeteroGraph,
                               List[deepsnap.hetero_graph.HeteroGraph]]:
    r"""Loads a single (homogenous/heterogeneous) graph or a list of graphs.

    Args:
        dataset_dir: the location of data on the disk.

        snapshot: load the entire dataset as a single graph or as a list
            of sub-graph snapshots.

        snapshot_freq: only used when dataset_dir is a single file.
            Split the entire it into snapshots of transactions with this
            provided snapshot_freq.
            E.g., the dataset_dir is a tsv file of all transactions in 2008,
            12 snapshot graphs of monthly transactions will be generated with
            snapshot_freq = 'M'.

        is_hetero: whether to load graphs as heterogeneous graphs.

    Returns:
        A single graph object if snapshot == False.
        A list of graph objects if snapshot == True.
    """
    # ==================== arg check ====================
    if not os.path.isfile(dataset_dir):
        raise ValueError(f'Provided dir {dataset_dir} is not a file.')

    if not os.path.exists(dataset_dir):
        raise ValueError(f'Provided dir {dataset_dir} does not exist.')

    if snapshot:
        if not isinstance(snapshot_freq, str):
            raise ValueError(f'snapshot_freq must be provided as a string, '
                             f'got: {type(snapshot_freq)} instead.')

        if snapshot_freq not in ['D', 'W', 'M', 'Q']:
            raise ValueError(f'Unsupported snapshot_freq: {snapshot_freq}')
    # ==================== arg check ends ====================
    # load the entire graph.
    g_all, node_type2id, edge_type2id, message_type2id = load_single_dataset(
        dataset_dir, is_hetero)
    # return the entire dataset as one single graph.
    if not snapshot:
        if is_hetero:
            return make_hetero_graph(g_all, node_type2id, message_type2id)
        else:
            return g_all

    # otherwise, split g_all into snapshots.
    snapshot_list = make_graph_snapshot(g_all, snapshot_freq,
                                        node_type2id=node_type2id,
                                        message_type2id=message_type2id,
                                        is_hetero=is_hetero)

    num_nodes = g_all.edge_index.max() + 1
    # init node degree.
    node_degree_existing = torch.zeros(num_nodes)
    for g_snapshot in snapshot_list:
        # init node stages (hidden representations) for each layer.
        if is_hetero:
            node_states_dict, node_degree_existing_dict = dict(), dict()
            for node_type in node_type2id.keys():
                node_states_dict[node_type] = [0 for _ in
                                               range(cfg.gnn.layers_mp)]

                node_degree_existing_dict[node_type] = torch.zeros(
                    g_snapshot.node_feature[node_type].shape[0])

            g_snapshot.node_states = node_states_dict
            g_snapshot.node_degree_existing = node_degree_existing_dict
        else:
            g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_degree_existing = node_degree_existing.clone()

    # Message types with too few edges cause problem during train-test split,
    # remove them.
    for graph in snapshot_list:
        all_types = list(graph.edge_index.keys())
        for key in all_types:
            if torch.numel(graph.edge_index[key]) / 2 <= 10:
                del graph.edge_index[key]
                del graph.edge_feature[key]
                del graph.edge_time[key]

    return snapshot_list


def make_id2type(type2id_dict: Dict[object, np.ndarray],
                 num_items: int) -> np.ndarray:
    """Makes the reversed mappings, outputs an array that maps index
    to type.
    Args:
        type2id_dict: Dict[object, array], keys are node/edge types
            and values are indices of nodes/edges belonging to this type.

        num_items: number of nodes/edges to be expected.

    Returns:
        output array of objects: output[i] = type of node/edge i.
    """
    id2type_ar = np.empty(num_items).astype(object)
    id2type_ar[:] = None

    for item_type, members in type2id_dict.items():
        id2type_ar[members] = item_type

    return id2type_ar


def load_dataset_transaction_bsi(format, name, dataset_dir):
    bsi_datasets = [
        'transactions_2008.tsv',
        'transactions_large.tsv',
        'transactions_svt_2008.tsv',
        'transactions_svt.tsv',
        'transactions_t2_2008.tsv',
        'transactions_t2.tsv',
        'transactions_zk_2008.tsv',
        'transactions_zk.tsv'
    ]
    if format == 'transaction_hetero_v1':
        if name in bsi_datasets:
            dataset_dir = os.path.join(dataset_dir, name)
            graphs = load_bsi_hetero(dataset_dir,
                                     snapshot=cfg.transaction.snapshot,
                                     snapshot_freq=cfg.transaction.snapshot_freq,
                                     is_hetero=cfg.dataset.is_hetero)
            return graphs


register_loader('roland_bsi_hetero', load_dataset_transaction_bsi)
