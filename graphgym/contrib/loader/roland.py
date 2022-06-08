import os
from datetime import datetime

import deepsnap
from deepsnap.dataset import GraphDataset
from torch_geometric.data import Data
from torch_geometric.datasets import *
import pandas as pd
import torch
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, \
    StandardScaler, MinMaxScaler
from deepsnap.graph import Graph

import pdb

from graphgym.contrib.loader.roland_bsi import  load_bsi

from graphgym.config import cfg
from graphgym.register import register_loader


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


def feature2time(df: pd.DataFrame, feature_name: str = 'Time_step',
                 format='torch'):
    df_time = pd.to_datetime(df[feature_name]).astype(int) / 6e10
    X = df_time.values.astype(float).reshape(-1, 1)
    enc = MinMaxScaler((0, 2))
    enc.fit(X)
    X = enc.transform(X)
    if format == 'torch':
        return torch.Tensor(X)
    else:
        return X


def copyempty(x):
    if x[1] == '  ':
        return x[0]
    else:
        return x[1]


def load_jpmc_new(dataset_dir, preprocessed=True, has_label=True,
                  snapshot=True, snapshot_num=100):
    # load data
    df_trans = pd.read_csv(dataset_dir + '.csv')
    # preprocess data
    if not preprocessed:
        df_trans.drop(['LOB', 'Name'], axis=1, inplace=True)
        df_trans.dropna(axis=0, how='any', inplace=True)
        df_trans['Bene_Id'] = df_trans[['Sender_Id', 'Bene_Id']].apply(
            lambda x: copyempty(x), axis=1)
        df_trans['Bene_Country'] = df_trans[
            ['Sender_Country', 'Bene_Country']].apply(lambda x: copyempty(x),
                                                      axis=1)
        df_trans['Sender_Id_type'] = df_trans['Sender_Id'].apply(
            lambda x: x.split('-')[0])
        df_trans['Bene_Id_type'] = df_trans['Bene_Id'].apply(
            lambda x: x.split('-')[0])

    # extract attributes to build the graph
    edge_feature1 = features2int(df_trans,
                                 features_name=['Sector', 'Transaction_Type'])
    edge_feature2 = features2int(df_trans,
                                 features_name=['Sender_Country',
                                                'Bene_Country'],
                                 concat=True)
    edge_feature3 = features2int(df_trans,
                                 features_name=['Sender_Id_type',
                                                'Bene_Id_type'],
                                 concat=True)
    edge_amount = features2float(df_trans, features_name=['USD_Amount'])
    edge_time = feature2time(df_trans, feature_name='Time_step')
    edge_feature = torch.cat((edge_feature1, edge_feature2, edge_feature3,
                              edge_amount, edge_time), dim=1)

    # todo: consider backward link
    edge_index = features2int(df_trans,
                              features_name=['Sender_Id', 'Bene_Id'],
                              concat=True).permute(1, 0).long()

    edge_label = torch.tensor(df_trans['Label'].values).long()

    n = edge_index.max() + 1
    node_feature = torch.zeros(n, 1)
    # init node features for each layer
    # node_states = [0 for i in range(cfg.gnn.layers_mp)]
    # init node degree
    # node_degree_existing = torch.zeros(n)

    # split into snapshots
    if snapshot:
        graphs = []
        snapshot_size = len(df_trans) // snapshot_num
        for i in range(snapshot_num):
            edge_feature_i = edge_feature[snapshot_size * i:snapshot_size * (i + 1)]
            edge_label_i = edge_label[snapshot_size * i:snapshot_size * (i + 1)]
            edge_index_i = edge_index[:, snapshot_size * i:snapshot_size * (i + 1)]
            edge_time_i = edge_time[snapshot_size * i:snapshot_size * (i + 1)]
            if has_label:
                graph = Graph(node_feature=node_feature,
                              edge_feature=edge_feature_i,
                              edge_label=edge_label_i,
                              edge_index=edge_index_i,
                              edge_time=edge_time_i,
                              # node_states=node_states,
                              node_states=[0 for _ in
                                           range(cfg.gnn.layers_mp)],
                              # node_degree_existing=node_degree_existing,
                              node_degree_existing=torch.zeros(n),
                              directed=True)
            else:
                graph = Graph(node_feature=node_feature,
                              edge_feature=edge_feature_i,
                              edge_index=edge_index_i,
                              edge_time=edge_time_i,
                              # node_states=node_states,
                              node_states=[0 for _ in
                                           range(cfg.gnn.layers_mp)],
                              # node_degree_existing=node_degree_existing,
                              node_degree_existing=torch.zeros(n),
                              directed=True)
            graphs.append(graph)
    else:
        graph = Graph(node_feature=node_feature, edge_feature=edge_feature,
                      edge_label=edge_label, edge_index=edge_index,
                      edge_time=edge_time,
                      # node_states=node_states,
                      node_states=[0 for _ in range(cfg.gnn.layers_mp)],
                      # node_degree_existing=node_degree_existing,
                      node_degree_existing=torch.zeros(n),
                      directed=True)
        graphs = [graph]


    return graphs


def load_dataset_transaction(format, name, dataset_dir):
    bsi_datasets = [
        "transactions_2008.tsv",
        "transactions_large.tsv",
        "transactions_svt_2008.tsv",
        "transactions_svt.tsv",
        "transactions_t2_2008.tsv",
        "transactions_t2.tsv",
        "transactions_zk_2008.tsv",
        "transactions_zk.tsv"
    ]
    if 'jpmc' in name:
        filename = 'jpmc'
    else:
        filename = name
    dataset_dir = os.path.join(dataset_dir, filename)
    if format == 'transaction':
        if name == 'jpmc':
            has_label = True if cfg.dataset.task == 'edge' else False
            graphs = load_jpmc_new(dataset_dir,
                                   has_label=has_label,
                                   snapshot=cfg.transaction.snapshot,
                                   snapshot_num=cfg.transaction.snapshot_num)
            return graphs



register_loader('roland', load_dataset_transaction)
