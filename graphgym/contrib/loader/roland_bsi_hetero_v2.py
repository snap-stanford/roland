"""
Another version of heterogeneous graph loader, however, this loader outputs
a plain homogenous Graph object, with node/edge type information appended
to node/edge features or as additional attributes to the Graph object.
Mar. 1, 2021
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
    mask_small = df['AmountEUR'] < 500
    mask_medium = np.logical_and(df['AmountEUR'] >= 500,
                                 df['AmountEUR'] < 1000)
    mask_large = df['AmountEUR'] >= 1000
    amount_level[mask_small] = '$<500'
    amount_level[mask_medium] = '500<=$<1k'
    amount_level[mask_large] = '$>=1k'

    df['AmountLevel'] = amount_level

    return df


def load_single_dataset(dataset_dir: str, is_hetero: bool = True,
                        node_type_defn: List[str] = ['OutOfCountry'],
                        edge_type_defn: List[str] = ['AmountLevel'],
                        type_info_loc: str = 'append'
                        ) -> Graph:
    """
    Loads a single graph object from tsv file.

    Args:
        dataset_dir: the path of tsv file to be loaded.

        is_hetero: whether to load heterogeneous graph.

        node_type_defn: a list of columns of the dataset used to define
            node types. Only necessary if is_hetero == True.

        edge_type_defn: a list of columns of the dataset used to define
            edge types. Only necessary if is_hetero == True.

        type_info_loc: 'append' or 'graph_attribute'.

    Returns:
        graph: a (homogenous) deepsnap graph object.
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

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )

    if is_hetero:
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

        node_type_enc = SkOrdinalEncoder()
        node_type_int = node_type_enc.fit_transform(
            df_char['NodeType'].values.reshape(-1, 1))
        node_type_int = torch.FloatTensor(node_type_int)
        assert len(node_type_int) == num_nodes

        # Construct edge type signatures.
        df_trans['EdgeType'] = df_trans[edge_type_defn[0]].astype(str)
        for var in edge_type_defn[1:]:
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

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed,
            list_n_type=g_all.list_n_type,
            list_e_type=g_all.list_e_type,
        )
        if is_hetero and hasattr(g_all, 'node_type'):
            g_incr.node_type = g_all.node_type
            g_incr.edge_type = g_all.edge_type[period_members]
        snapshot_list.append(g_incr)
    return snapshot_list


def load_bsi_hetero(dataset_dir: str,
                    snapshot: bool = True,
                    snapshot_freq: str = None,
                    is_hetero: bool = True,
                    type_info_loc: str = 'append'
                    ) -> Union[deepsnap.graph.Graph,
                               List[deepsnap.graph.Graph]]:
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
    # load the entire graph.
    g_all = load_single_dataset(dataset_dir, is_hetero=is_hetero,
                                type_info_loc=type_info_loc)
    # return the entire dataset as one single graph.
    if not snapshot:
        return g_all

    # otherwise, split g_all into snapshots.
    snapshot_list = make_graph_snapshot(g_all, snapshot_freq, is_hetero)

    num_nodes = g_all.edge_index.max() + 1
    # init node degree.
    for g_snapshot in snapshot_list:
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        g_snapshot.node_degree_existing = torch.zeros(num_nodes)

    return snapshot_list


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
    if format == 'transaction_hetero_v2':
        if name in bsi_datasets:
            dataset_dir = os.path.join(dataset_dir, name)
            graphs = load_bsi_hetero(dataset_dir,
                                     snapshot=cfg.transaction.snapshot,
                                     snapshot_freq=cfg.transaction.snapshot_freq,
                                     is_hetero=cfg.dataset.is_hetero,
                                     type_info_loc=cfg.dataset.type_info_loc)
            return graphs


register_loader('roland_bsi_hetero_v2', load_dataset_transaction_bsi)
