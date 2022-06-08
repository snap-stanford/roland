import graphgym.register as register
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from graphgym.config import cfg
from graphgym.contrib.stage import *
from graphgym.init import init_weights
from graphgym.models.act import act_dict
from graphgym.models.feature_augment import Preprocess
from graphgym.models.feature_encoder import (edge_encoder_dict,
                                             node_encoder_dict)
from graphgym.models.head import head_dict
from graphgym.models.layer import (BatchNorm1dEdge, BatchNorm1dNode,
                                   GeneralLayer, GeneralMultiLayer,
                                   GeneralRecurrentLayer,
                                   GraphRecurrentLayerWrapper,
                                   GRUGraphRecurrentLayer, layer_dict)
from graphgym.register import register_network


class StaticGNN(torch.nn.Module):
    """The very basic GCN used for benchmarking purpose."""
    def __init__(self, dim_in, dim_out, num_layers=1, model_type='GCN', layer_id=0):
        super(StaticGNN, self).__init__()
        conv_model = self.build_conv_model(model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(dim_in, dim_in))
        self.layer_id = layer_id
        self.num_layers = num_layers
        for _ in range(num_layers - 1):
            self.convs.append(conv_model(dim_in, dim_in))

        GNNHead = head_dict[cfg.dataset.task]
        self.post_mp = GNNHead(dim_in=dim_in, dim_out=dim_out)
        self.dropout = cfg.gnn.dropout

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GAT':
            return pyg_nn.GATConv
        elif model_type == "GraphSage":
            return pyg_nn.SAGEConv
        else:
            raise ValueError(
                "Model {} unavailable".format(model_type))

    def forward(self, batch):
        x, edge_index = batch.node_feature, batch.edge_index 

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        batch.node_feature = x
        # don't do any memory thing, moving average is disabled.
        batch.node_states[self.layer_id] = x.clone()
        return batch


def GNNLayer(dim_in, dim_out, has_act=True, id=0):
    return StaticGNN(dim_in, dim_out, num_layers=1, layer_id=id)


def GNNPreMP(dim_in, dim_out):
    r"""Constructs preprocessing layers: dim_in --> dim_out --> dim_out --> ... --> dim_out"""
    return GeneralMultiLayer('linear', cfg.gnn.layers_pre_mp,
                             dim_in, dim_out, dim_inner=dim_out,
                             final_act=True)


class GNNStackStage(nn.Module):
    r"""Simple Stage that stacks GNN layers"""

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNStackStage, self).__init__()
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out, id=i)
            self.add_module('layer{}'.format(i), layer)
        self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


stage_dict = {'stack': GNNStackStage}

stage_dict = {**register.stage_dict, **stage_dict}


########### Model: start + stage + head ############

class GNN(nn.Module):
    r"""The General GNN model"""

    def __init__(self, dim_in, dim_out, **kwargs):
        r"""Initializes the GNN model.

        Args:
            dim_in, dim_out: dimensions of in and out channels.
            Parameters:
            node_encoding_classes - For integer features, gives the number
            of possible integer features to map.
        """
        super(GNN, self).__init__()
        GNNStage = stage_dict[cfg.gnn.stage_type]
        GNNHead = head_dict[cfg.dataset.task]
        # Currently only for OGB datasets
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = node_encoder_dict[cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(cfg.dataset.encoder_dim)
            # Update dim_in to reflect the new dimension fo the node features
            dim_in = cfg.dataset.encoder_dim
        if cfg.dataset.edge_encoder:
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = edge_encoder_dict[cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.dataset.encoder_dim)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dEdge(cfg.dataset.edge_dim)
        self.preprocess = Preprocess(dim_in)
        d_in = self.preprocess.dim_out
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(d_in, cfg.gnn.dim_inner)
            d_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp >= 1:
            self.mp = GNNStage(dim_in=d_in,
                               dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
            d_in = self.mp.dim_out
        self.post_mp = GNNHead(dim_in=d_in, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


register_network('gnn_static', GNN)
