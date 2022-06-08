"""
This general framework adapts models from other recurrent graph neural net
papers to the graphgym framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import graphgym.register as register
from graphgym.config import cfg
from graphgym.init import init_weights
from graphgym.models.act import act_dict
from graphgym.models.head import head_dict
from graphgym.models.layer import (GeneralMultiLayer,
                                   layer_dict)
from graphgym.register import register_network

recurrent_layer_types = [
    'dcrnn', 'evolve_gcn_o', 'evolve_gcn_h', 'gconv_gru', 'gconv_lstm',
    'gconv_lstm_baseline', 'tgcn', 'edge_conv_gru'
]


def GNNLayer(dim_in, dim_out, has_act=True, id=0):
    assert cfg.gnn.layer_type in recurrent_layer_types
    return layer_dict[cfg.gnn.layer_type](dim_in, dim_out, id=id)


def GNNPreMP(dim_in, dim_out):
    r"""
    Constructs preprocessing layers:
        dim_in --> dim_out --> dim_out --> ... --> dim_out
    """
    return GeneralMultiLayer('linear', cfg.gnn.layers_pre_mp,
                             dim_in, dim_out, dim_inner=dim_out,
                             final_act=True)


class GNNSkipBlock(nn.Module):
    '''Skip block for GNN'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipBlock, self).__init__()
        if num_layers == 1:
            self.f = [GNNLayer(dim_in, dim_out, has_act=False)]
        else:
            self.f = []
            for i in range(num_layers - 1):
                d_in = dim_in if i == 0 else dim_out
                self.f.append(GNNLayer(d_in, dim_out))
            d_in = dim_in if num_layers == 1 else dim_out
            self.f.append(GNNLayer(d_in, dim_out, has_act=False))
        self.f = nn.Sequential(*self.f)
        self.act = act_dict[cfg.gnn.act]
        if cfg.gnn.stage_type == 'skipsum':
            assert dim_in == dim_out, 'Sum skip must have same dim_in, dim_out'

    def forward(self, batch):
        node_feature = batch.node_feature
        if cfg.gnn.stage_type == 'skipsum':
            batch.node_feature = \
                node_feature + self.f(batch).node_feature
        elif cfg.gnn.stage_type == 'skipconcat':
            batch.node_feature = \
                torch.cat((node_feature, self.f(batch).node_feature), 1)
        else:
            raise ValueError(
                'cfg.gnn.stage_type must in [skipsum, skipconcat]')
        batch.node_feature = self.act(batch.node_feature)
        return batch


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


class GNNSkipStage(nn.Module):
    ''' Stage with skip connections'''

    def __init__(self, dim_in, dim_out, num_layers):
        super(GNNSkipStage, self).__init__()
        assert num_layers % cfg.gnn.skip_every == 0, \
            'cfg.gnn.skip_every must be multiples of cfg.gnn.layer_mp' \
            '(excluding head layer)'
        for i in range(num_layers // cfg.gnn.skip_every):
            if cfg.gnn.stage_type == 'skipsum':
                d_in = dim_in if i == 0 else dim_out
            elif cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            block = GNNSkipBlock(d_in, dim_out, cfg.gnn.skip_every)
            self.add_module('block{}'.format(i), block)
        if cfg.gnn.stage_type == 'skipconcat':
            self.dim_out = d_in + dim_out
        else:
            self.dim_out = dim_out

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        if cfg.gnn.l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=-1)
        return batch


stage_dict = {
    'stack': GNNStackStage,
    'skipsum': GNNSkipStage,
    'skipconcat': GNNSkipStage,
}

stage_dict = {**register.stage_dict, **stage_dict}


class EdgeFeature2WeightLayer(nn.Module):
    # Construct edge weights from edge features.
    def __init__(self, edge_dim: int):
        super(EdgeFeature2WeightLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(edge_dim, 1, bias=True),
            nn.Sigmoid(),
            nn.Linear(1, 1, bias=False)
        )

    def forward(self, batch):
        batch.edge_weight = self.layer(batch.edge_feature).view(-1, )
        return batch


class GNNBaseline(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(GNNBaseline, self).__init__()
        GNNStage = stage_dict[cfg.gnn.stage_type]
        GNNHead = head_dict[cfg.dataset.task]

        # Baseline models does not contain {node, edge} feature encoders and
        # batch-norm.
        # TODO: still need to check with the original implementation.

        # cfg.dataset.edge_dim # Need to be given explicitly

        d_in = dim_in
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(d_in, cfg.gnn.dim_inner)
            d_in = cfg.gnn.dim_inner

        # most baseline models does not use edge_feature but they take edge
        # weights. Optionally, we can map edge_feature to edge_weight for
        # these models for fair comparison.
        self.edge_feature2weight = EdgeFeature2WeightLayer(
            edge_dim=cfg.dataset.edge_dim)

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


register_network('gnn_recurrent_baseline', GNNBaseline)
