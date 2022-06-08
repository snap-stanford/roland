"""
This script contains method to load bsi dataset, load the entire dataset
as a single graph or a list of graph snapshots.
Jan. 17, 2021
"""
import os
from datetime import datetime
from glob import glob
from typing import List, Union

import deepsnap
import numpy as np
import pandas as pd
import torch
from deepsnap.graph import Graph
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

from graphgym.config import cfg
from graphgym.register import register_loader


def timeit(method):
    def timed(*args, **kw):
        ts = datetime.now()
        result = method(*args, **kw)
        te = datetime.now()
        print(f'{method.__name__} takes {te - ts}')
        return result
    return timed


@timeit
def features2int(df: pd.DataFrame, features_name: List = ['Sector'],
                 concat=False, format='torch'):
    X = df[features_name].values.astype(str)
    enc = OrdinalEncoder()
    if concat:
        enc.fit(np.concatenate((X, X[:, ::-1]), axis=0))
    else:
        enc.fit(X)
    X = enc.transform(X)
    if format == 'torch':
        return torch.Tensor(X)
    else:
        return X


@timeit
def features2float(df: pd.DataFrame, features_name: List = ['Sector'],
                   concat=False, format='torch'):
    X = df[features_name].values.astype(float)
    enc = MinMaxScaler((0, 2))
    if concat:
        enc.fit(np.concatenate((X, X[:, ::-1]), axis=0))
    else:
        enc.fit(X)
    X = enc.transform(X)
    if format == 'torch':
        return torch.Tensor(X)
    else:
        return X


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


def __load_single_dataset__(
        dataset_dir: str,
        preprocessed: bool = False,
        skd_split: bool = True,
        skis_split: bool = True,
        include_node_features: bool = False
) -> Graph:
    """
    The core data loader for BSI dataset, only load single dataset.
    Args:
        dataset_dir: a string representing the directory of
            transaction dataset.

        preprocessed: whether the dataset has been preprocessed. If not,
            the loader preprocess the dataset
            immediately after loading it from the disk.

        skd_split: boolean indicating how this method loads SKD of payers and
                payees.
            SkdL1 + SkdL2 provide a complete description of Skd.
            if skd_split == True:
                Ordinal encode SkdL1 and SkdL2 separately and save them as
                distinct features.
            if skd_split == False:
                Ordinal encode Skd directly, this option leads to one
                feature but with much larger dimension.

        skis_split: see skd_split above.

        include_node_features: whether to include node features.
            Please note that node features have already been included in edge
            features as {Payer, Payee}_attr.
    Returns:
        A deepsnap graph object.

    Examples:
        dataset_dir = "/lfs/hyperturing2/0/tianyudu/bsi/all/data
        /transactions_t2.tsv"
        dataset_dir = "/lfs/hyperturing2/0/tianyudu/bsi/all/data
        /transactions_svt_2008.tsv"
    """
    # Load data from disk
    # TODO: load large dataset, such as ZK set, using dask.
    print('Loading data from disk....')
    t0 = datetime.now()
    df_trans = pd.read_csv(dataset_dir, sep="\t",
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
                                  'DayOfMonth': np.int32, 'DayOfWeek': np.int32,
                                  'Timestamp': np.int64},
                           low_memory=False)
    t1 = datetime.now()
    print(f'Data loading takes {t1 - t0}')
    # Preprocess data if needed.
    if not preprocessed:
        df_trans.fillna('missing', inplace=True)
    # Extract attributes to build the graph's edge features.
    # ef stands for edge_feature.
    ef_bank = features2int(df_trans, features_name=['PayerBank', 'PayeeBank'],
                           concat=True)
    ef_country = features2int(df_trans,
                              features_name=['PayerCountry', 'PayeeCountry'],
                              concat=True)
    ef_region = features2int(df_trans,
                             features_name=['PayerRegion', 'PayeeRegion'],
                             concat=True)

    if skd_split:
        # Encode L1 and L2 separately and treat them as distinct features.
        ef_skdl1 = features2int(df_trans,
                                features_name=['PayerSkdL1', 'PayeeSkdL1'],
                                concat=True)
        ef_skdl2 = features2int(df_trans,
                                features_name=['PayerSkdL2', 'PayeeSkdL2'],
                                concat=True)
        ef_skd = torch.cat((ef_skdl1, ef_skdl2), dim=1)
    else:
        # Encode entire Skd at once, lead to one single feature.
        ef_skd = features2int(df_trans, features_name=['PayerSkd', 'PayeeSkd'],
                              concat=True)

    if skis_split:
        ef_skisl1 = features2int(df_trans,
                                 features_name=['PayerSkisL1', 'PayeeSkisL1'],
                                 concat=True)
        ef_skisl2 = features2int(df_trans,
                                 features_name=['PayerSkisL2', 'PayeeSkisL2'],
                                 concat=True)
        ef_skis = torch.cat((ef_skisl1, ef_skisl2), dim=1)
    else:
        ef_skis = features2int(df_trans,
                               features_name=['PayerSkis', 'PayeeSkis'],
                               concat=True)

    # Add features of transaction.
    ef_categorical = features2int(df_trans,
                                  features_name=['# System', 'Currency'])
    edge_amount = features2float(df_trans,
                                 features_name=['AmountEUR'])
    # There is another `Amount` attribute, but we cares about the EUR one.

    edge_time = torch.FloatTensor(df_trans['Timestamp'])  # (E,)

    # edge_time = df_trans['Timestamp'].values.reshape(-1, 1)
    # edge_time = edge_time / 6e10
    # enc = MinMaxScaler((0, 2))
    # edge_time_scaled = torch.Tensor(enc.fit_transform(
    #     edge_time.reshape(-1, 1)))
    edge_time_scaled = features2float(df_trans,
                                      features_name=['Timestamp'])
    # Combine edge features.
    edge_feature = torch.cat((ef_bank, ef_country, ef_region, ef_skd, ef_skis,
                              ef_categorical, edge_amount,
                              edge_time_scaled.view(-1, 1)),
                             dim=1)

    edge_index = features2int(df_trans,
                              features_name=['Payer', 'Payee'], concat=True
                              ).permute(1, 0).long()
    # the same as transpose(*, 0, 1)

    num_nodes = torch.max(edge_index) + 1

    if include_node_features:
        raise NotImplementedError
    else:
        node_feature = torch.ones(num_nodes, 1).float()

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )
    return graph


def __subset_graph_edges__(graph, mask):
    r"""A helper function, get the induced subgraph by an edge_mask."""
    g_subset = Graph(
        node_feature=graph.node_feature,
        edge_feature=graph.edge_feature[mask, :],
        edge_index=graph.edge_index[:, mask],
        edge_time=graph.edge_time[mask],
        directed=graph.directed
    )
    return g_subset


def load_bsi(
        dataset_dir: str,
        snapshot: bool = True,
        snapshot_freq: str = None,
        preprocessed: bool = False,
        skd_split: bool = True,
        skis_split: bool = True,
        include_node_features: bool = False,
) -> Union[deepsnap.graph.Graph, List[deepsnap.graph.Graph]]:
    r"""Loads a sequence of graph snapshots.

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

        # Args passed into load_bsi() method, refer to load_bsi's docstring.
        preprocessed:
        skd_split:
        skis_split:
        include_node_features:

    Returns:
        A single graph object if snapshot == False.
        A list of graph objects if snapshot == True.
    """
    # A list of Graph objects, each represents a snapshot, snapshots
    # follow the chronological order.
    snapshot_list = list()

    # ==================== arg check ====================
    if not os.path.isfile(dataset_dir):
        raise ValueError(f'Provided dir {dataset_dir} is not a file.')

    if not os.path.exists(dataset_dir):
        raise ValueError(f'Provided dir {dataset_dir} does not exist.')

    if snapshot:
        if not isinstance(snapshot_freq, str):
            raise ValueError(f'snapshot_freq must be provided as a string, '
                             f'got: {type(snapshot_freq)} instead.')

        if snapshot_freq not in ['D', 'W', 'M', 'Q', 'Y']:
            raise ValueError(f'Unsupported snapshot_freq: {snapshot_freq}')
    # ==================== arg check ends ====================
    # load the entire graph.
    g_all = __load_single_dataset__(dataset_dir, preprocessed, skd_split,
                                    skis_split, include_node_features)
    # return the entire dataset as one single graph.
    if not snapshot:
        return g_all

    # otherwise, split g_all into snapshots.
    # edge_timestamp integers.
    edge_ts = pd.Series(g_all.edge_time).astype(np.int64)

    # get year of a particular timestamp.
    def get_yr(ts: int) -> int:
        return int(datetime.fromtimestamp(ts).strftime('%Y'))

    edge_yr = edge_ts.apply(get_yr).astype(np.int64).values  # @(E,)
    years = list(set(edge_yr))
    years.sort()  # ascending order.
    for yr in years:
        # graph of the current year.
        yr_e_mask = (edge_yr == yr)  # @(E,)
        g_yr = __subset_graph_edges__(g_all, yr_e_mask)
        if snapshot_freq == 'Y':
            # For yearly frequency,
            # add the entire yearly graph as snapshot.
            snapshot_list.append(g_yr)
        else:
            # for other finer frequencies, further split the graph.
            # datetime associated with transaction edge in this year.
            t = edge_ts[yr_e_mask].astype(np.int64)
            # sanity check
            assert len(t) == len(g_yr.edge_time) == len(g_yr.edge_time)
            assert torch.all(torch.Tensor(t.values) == g_yr.edge_time)

            # Construct flags for splitting dataset, split graph edges
            # according to values in ind_flag.
            if snapshot_freq == 'D':
                ind_flag = t.apply(dayofyear)
            elif snapshot_freq == 'W':
                ind_flag = t.apply(weekofyear)
            elif snapshot_freq == 'M':
                ind_flag = t.apply(monthofyear)
            elif snapshot_freq == 'Q':
                ind_flag = t.apply(quarterofyear)
            else:
                raise NotImplementedError

            # improve performance, pd.Series --> np.ndarray.
            ind_flag = ind_flag.values
            # len(periods) == number snapshots this year.
            periods = list(set(ind_flag))
            periods.sort()  # ascending order.
            for p in periods:
                # bool indicates edges in the current period.
                period_e_mask = (ind_flag == p)  # @(E_yr,)
                g_incr = __subset_graph_edges__(g_yr, period_e_mask)
                snapshot_list.append(g_incr)

    n = g_all.edge_index.max() + 1
    for g_snapshot in snapshot_list:
        # init node stages (hidden representations) for each layer.
        g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
        # init node degree.
        g_snapshot.node_degree_existing = torch.zeros(n)

    # sort snapshot by the edge timestamp, enforce the chronological order.
    snapshot_list.sort(key=lambda g: int(g.edge_time.numpy()[0]))
    if cfg.transaction.check_snapshot:
        __check_snapshot__(snapshot_list, snapshot_freq, True)
    return snapshot_list


def __check_snapshot__(
    snapshot_list: List,
    snapshot_freq: str,
    verbose: bool
) -> None:
    r"""
    A helper function to check the chronological ordering of snapshots.
    """
    if verbose:
        print('=' * 80)
        print(f'Verify loaded {len(snapshot_list)} '
              f'snapshot/incremental graphs.')

    # global start and end dates.
    start_date = np.min([g.edge_time.min().numpy() for g in snapshot_list])
    end_date = np.max([g.edge_time.max().numpy() for g in snapshot_list])
    assert start_date <= end_date

    def ts2dt(ts):  # timestamp2datetime
        return pd.to_datetime(ts, unit='s')

    if verbose:
        print(f"Transactions range from {ts2dt(start_date)} "
              f"to {ts2dt(end_date)}")

    d = {'idx': [],
         'start': [], 'sDoY': [], 'sWoY': [],
         'end': [], 'eDoY': [], 'eWoY': [],
         '|V|': [], '|E|': [], 'repr': []}

    prev_end = 0
    for i, g_incr in enumerate(snapshot_list):
        # edges should be sorted based on their time.
        assert np.all(np.diff(g_incr.edge_time) >= 0)
        # ensure chronological stepping.
        assert g_incr.edge_time.numpy()[0]\
               == g_incr.edge_time.numpy().min()

        assert g_incr.edge_time.numpy()[-1]\
               == g_incr.edge_time.numpy().max()

        cur_start = int(g_incr.edge_time.numpy()[0])
        cur_end = int(g_incr.edge_time.numpy()[-1])

        assert prev_end <= cur_start
        assert cur_start <= cur_end

        if snapshot_freq is not None:
            if snapshot_freq == 'D':
                s = dayofyear(cur_start)
                e = dayofyear(cur_end)
            elif snapshot_freq == 'W':
                s = weekofyear(cur_start)
                e = weekofyear(cur_end)
            elif snapshot_freq == 'M':
                s = monthofyear(cur_start)
                e = monthofyear(cur_end)
            elif snapshot_freq == 'Q':
                s = quarterofyear(cur_start)
                e = quarterofyear(cur_end)
            assert s == e, f'received {s} != {e}'

        # Prepare for reporting.
        if verbose:

            d['idx'].append(i+1)

            d['start'].append(ts2dt(cur_start))
            d['sDoY'].append(dayofyear(cur_start))
            d['sWoY'].append(weekofyear(cur_start))

            d['end'].append(ts2dt(cur_end))
            d['eDoY'].append(dayofyear(cur_end))
            d['eWoY'].append(weekofyear(cur_end))

            d['|V|'].append(g_incr.num_nodes)
            d['|E|'].append(g_incr.num_edges)
            # d['repr'].append(repr(g_incr))
            d['repr'].append(None)

        prev_end = cur_end

    if verbose:
        print('Summary of graph snapshots')
        df = pd.DataFrame(d)
        print(df)

    print('Snapshot data check: ' + '\x1b[6;30;42m' + 'Passed!' + '\x1b[0m')
    if verbose:
        print('=' * 80)


def save_bsi_to_tensor(graphs, tensor_dir) -> None:
    if not cfg.transaction.snapshot:
        # Save single graph to the target directory.
        p = os.path.join(tensor_dir, 'graph.pt')
        torch.save(graphs, p)
    else:
        # List of graph snapshots.
        for i, graph in enumerate(graphs):
            # make directory for each graph snapshot.
            # e.g., all/data/transactions_svt_2008_D/graph10.pt/
            p = os.path.join(tensor_dir, f'graph{i}.pt')
            torch.save(graph, p)


def load_bsi_from_tensor(tensor_dir):
    """Load pre-made graph snapshots from disk directly.
    """
    if not cfg.transaction.snapshot:
        # load single graph.
        return torch.load(os.path.join(tensor_dir, 'graph.pt'))
    else:
        # load snapshots.
        graph_snapshots = list()
        num_snapshots = len(glob(os.path.join(tensor_dir, 'graph*')))
        print(f'Loading {num_snapshots} from disk.')
        for i in range(num_snapshots):
            # Load individual snapshot.
            p = os.path.join(tensor_dir, f'graph{i}.pt')
            gs = torch.load(p)
            graph_snapshots.append(gs)

        return graph_snapshots


def __check_bsi_tensor__(fresh, tensor_dir, graph_attrs):
    """Check if graph tensors saved on disk are correct by comparing
    graphs just generated by data processing pipeline and the copy saved to
    disk. graph_attrs provides a list of attributes to be checked.
    """
    canned = load_bsi_from_tensor(tensor_dir)
    print('\x1b[6;30;42mCheck attributes '
          + ','.join(graph_attrs) + ' \x1b[0m')

    def compare(d1, d2):
        # Check each key attribute of two graphs (d1, d2).
        for k in graph_attrs:
            left = getattr(d1, k)
            right = getattr(d2, k)
            if isinstance(left, torch.Tensor):
                if not torch.all(left == right):
                    print('\x1b[0;31m' + f'attribute {k} not match, received:'
                          + '\x1b[0;31m')
                    print(left, right)
            else:
                if not left == right:
                    print('\x1b[0;31m' + f'attribute {k} not match, received:'
                          + '\x1b[0;31m')
                    print(left, right)

    if cfg.transaction.snapshot:
        # compare each snapshot.
        for (d1, d2) in zip(fresh, canned):
            compare(d1, d2)
    else:
        # compare the entire graph.
        compare(fresh, canned)

    print('Tensor integrity check: ' + '\x1b[6;30;42m' + 'Passed!' + '\x1b[0m')


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
    if format == 'transaction':
        if name in bsi_datasets:
            # The expected location of cached graphs on disk.
            # expected_pickle_path = f'{cfg.dataset.dir}/{name[:-4]}_' \
            #                        f'{cfg.transaction.snapshot_freq}'
            # print(f'Expected pickle path: {expected_pickle_path}')
            # if os.path.exists(expected_pickle_path):
            #     # load pickle from disk if they are cached results on disk.
            #     print('\x1b[0;31m'
            #           + f'Cached graphs found at {expected_pickle_path}, '
            #             f'load pickled snapshot graphs.'
            #           + '\x1b[0m')
            #     t0 = datetime.now()
            #     graphs = load_bsi_from_tensor(expected_pickle_path)
            #     t1 = datetime.now()
            #     print(f'Pickle loading time: {t1 - t0}')
            #     __check_snapshot__(graphs, cfg.transaction.snapshot_freq, True)
            #     t2 = datetime.now()
            #     print(f'Snapshot checking time: {t2 - t1}')
            # else:
            # print('No cached graphs found, load from tsv file.')
            # load tsv from disk.
            dataset_dir = os.path.join(dataset_dir, name)

            # t0 = datetime.now()

            graphs = load_bsi(dataset_dir,
                              snapshot=cfg.transaction.snapshot,
                              snapshot_freq=cfg.transaction.snapshot_freq)

                # t1 = datetime.now()
                # print(f'tsv loading + processing time: {t1 - t0}')
                #
                # # save a copy to local.
                # try:
                #     os.mkdir(expected_pickle_path)
                #     save_bsi_to_tensor(graphs, expected_pickle_path)
                #     print('\x1b[0;31m'
                #           + f'Cache graph tensors to {expected_pickle_path}'
                #           + '\x1b[0m')
                #
                #     t2 = datetime.now()
                #     print(f'Data saving time: {t2 - t1}')
                #
                #     # Validate integrity of saved pickle files.
                #     print('Validate cached graphs.')
                #     __check_bsi_tensor__(graphs, expected_pickle_path,
                #                          list(graphs[0].__dict__.keys()))
                #     t3 = datetime.now()
                #     print(f'Data checking time: {t3 - t2}')
                # except FileExistsError:
                #     # If a copy already exist on disk, skip.
                #     print('\x1b[0;31m'
                #           + f'Cache {expected_pickle_path} already exist, '
                #             f'will NOT save cache.'
                #           + '\x1b[0m')
            return graphs


register_loader('roland_bsi', load_dataset_transaction_bsi)

