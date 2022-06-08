import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

from graphgym.config import cfg
from graphgym.models.act import act_dict
from graphgym.contrib.layer.generalconv import (GeneralConvLayer,
                                                GeneralEdgeConvLayer)

from graphgym.contrib.layer import *
import graphgym.register as register

import deepsnap
import pdb

# General classes


class GeneralLayer(nn.Module):
    r"""General wrapper for layers that automatically constructs the
    learnable layer (e.g., graph convolution),
        and adds optional post-layer operations such as
            - batch normalization,
            - dropout,
            - activation functions.

    Note that this general layer only handle node features.
    """

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn and cfg.gnn.batchnorm
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            # Only modify node features here.
            batch.node_feature = self.post_layer(batch.node_feature)
            if self.has_l2norm:
                batch.node_feature = F.normalize(batch.node_feature, p=2,
                                                 dim=1)
        return batch


# General classes
class GeneralRecurrentLayer(nn.Module):
    r"""General wrapper for layers that automatically constructs the
    learnable layer (e.g., graph convolution),
        and adds optional post-layer operations such as
            - batch normalization,
            - dropout,
            - activation functions.

    Note that this general layer only handle node features.
    """

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, id=0, **kwargs):
        super(GeneralRecurrentLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.id = id
        has_bn = has_bn and cfg.gnn.batchnorm
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.layer(batch)
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        elif isinstance(batch.node_feature, dict):
            # Heterogeneous GNN.
            batch = self.layer(batch)
            for key in batch.node_feature.keys():
                # apply the same operations on every node type.
                batch.node_feature[key] = self.post_layer(
                    batch.node_feature[key])
                if self.has_l2norm:
                    batch.node_feature[key] = F.normalize(
                        batch.node_feature[key], p=2, dim=1)
                # weighted sum of new emb and old embedding
                batch.node_states[key][self.id] = \
                    batch.node_states[key][self.id] * batch.keep_ratio[key] \
                    + batch.node_feature[key] * (1 - batch.keep_ratio[key])
                batch.node_feature[key] = batch.node_states[key][self.id]
        else:
            # # Only modify node features here.
            # if self.id == 0:
            #     node_feature_input = batch.node_feature
            # else:
            #     node_feature_input = batch.node_states[self.id - 1]
            # # weighted sum of new emb and old embedding
            # if self.id == len(batch.node_states):
            #     # the final layer, output to head function
            #     batch.node_feature = \
            #         batch.node_states[self.id] * batch.keep_ratio \
            #         + self.post_layer(node_feature_input) * (
            #                     1 - batch.keep_ratio)
            # else:
            #     batch.node_states[self.id] = \
            #         batch.node_states[self.id] * batch.keep_ratio \
            #         + self.post_layer(node_feature_input) * (
            #                 1 - batch.keep_ratio)
            # Only modify node features here.
            # output to batch.node_feature
            # if torch.is_tensor(batch.node_states[self.id - 1]):
            #     print('before', batch.node_feature.sum(), batch.node_states[self.id - 1].sum())
            batch = self.layer(batch)
            # if torch.is_tensor(batch.node_states[self.id - 1]):
            #     print('after', batch.node_feature.sum(), batch.node_states[self.id - 1].sum())
            batch.node_feature = self.post_layer(batch.node_feature)
            if self.has_l2norm:
                batch.node_feature = F.normalize(batch.node_feature, p=2,
                                                 dim=1)
            # weighted sum of new emb and old embedding
            batch.node_states[self.id] = \
                batch.node_states[self.id] * batch.keep_ratio \
                + batch.node_feature * (1 - batch.keep_ratio)
            batch.node_feature = batch.node_states[self.id]
            # print('verify', batch.node_feature.sum(),
            #       batch.node_states[self.id].sum())

        return batch


class GRUGraphRecurrentLayer(nn.Module):
    r"""General wrapper for layers that automatically constructs the
    learnable layer (e.g., graph convolution),
        and adds optional post-layer operations such as
            - batch normalization,
            - dropout,
            - activation functions.

    This module updates nodes' hidden states differently based on nodes'
    activities. For nodes with edges in the current snapshot, their states
    are updated using an internal GRU; for other inactive nodes, their
    states are updated using simple MLP.
    """

    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, id=0, **kwargs):
        super(GRUGraphRecurrentLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer_id = id
        has_bn = has_bn and cfg.gnn.batchnorm
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)

        # dim_out = dim_hidden.
        # update gate.
        self.GRU_Z = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # reset gate.
        self.GRU_R = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # new embedding gate.
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Tanh())

        # self.direct_forward = nn.Sequential(
        #     nn.Linear(dim_in + dim_out, dim_out),
        #     nn.ReLU(),
        #     nn.Linear(dim_out, dim_out))

    def _init_hidden_state(self, batch):
        # Initialize hidden states of all nodes to zero.
        if not isinstance(batch.node_states[self.layer_id], torch.Tensor):
            batch.node_states[self.layer_id] = torch.zeros(
                batch.node_feature.shape[0], self.dim_out).to(
                batch.node_feature.device)

    def forward(self, batch):
        batch = self.layer(batch)
        batch.node_feature = self.post_layer(batch.node_feature)
        if self.has_l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=1)

        self._init_hidden_state(batch)
        # Compute output from GRU module.
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_gru = Z * H_prev + (1 - Z) * H_tilde

        if cfg.gnn.embed_update_method == 'masked_gru':
            # Update for active nodes only, use output from GRU.
            keep_mask = (batch.node_degree_new == 0)
            H_out = H_gru
            # Reset inactive nodes' embedding.
            H_out[keep_mask, :] = H_prev[keep_mask, :]
        elif cfg.gnn.embed_update_method == 'moving_average_gru':
            # Only update for active nodes, using moving average with output from GRU.
            H_out = H_prev * batch.keep_ratio + H_gru * (1 - batch.keep_ratio)
        elif cfg.gnn.embed_update_method == 'gru':
            # Update all nodes' embedding using output from GRU.
            H_out = H_gru
        else:
            raise ValueError(f'Invalid embedding update rule: {cfg.gnn.embed_update_method}')

        batch.node_states[self.layer_id] = H_out
        batch.node_feature = batch.node_states[self.layer_id]
        return batch


class GeneralMultiLayer(nn.Module):
    r"""General wrapper for stack of layers"""

    def __init__(self, name, num_layers, dim_in, dim_out, dim_inner=None,
                 final_act=True, **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(name, d_in, d_out, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch


# Core basic layers
# Input: batch; Output: batch
class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(Linear, self).__init__()
        self.model = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch


class BatchNorm1dNode(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dNode, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.node_feature = self.bn(batch.node_feature)
        return batch


class BatchNorm1dEdge(nn.Module):
    '''General wrapper for layers'''

    def __init__(self, dim_in):
        super(BatchNorm1dEdge, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in, eps=cfg.bn.eps, momentum=cfg.bn.mom)

    def forward(self, batch):
        batch.edge_feature = self.bn(batch.edge_feature)
        return batch


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, dim_inner=None,
                 num_layers=2, **kwargs):
        '''
        Note: MLP works for 0 layers
        '''
        super(MLP, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        layers = []
        if num_layers > 1:
            layers.append(
                GeneralMultiLayer('linear', num_layers - 1, dim_in, dim_inner,
                                  dim_inner, final_act=True))
            layers.append(Linear(dim_inner, dim_out, bias))
        else:
            layers.append(Linear(dim_in, dim_out, bias))
        self.model = nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.node_feature = self.model(batch.node_feature)
        return batch


class GCNConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GCNConv, self).__init__()
        self.model = pyg.nn.GCNConv(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class SAGEConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEConv, self).__init__()
        self.model = pyg.nn.SAGEConv(dim_in, dim_out, bias=bias, concat=True)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class GATConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATConv, self).__init__()
        self.model = pyg.nn.GATConv(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class GINConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GINConv, self).__init__()
        gin_nn = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(),
                               nn.Linear(dim_out, dim_out))
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class SplineConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SplineConv, self).__init__()
        self.model = pyg.nn.SplineConv(dim_in, dim_out,
                                       dim=1, kernel_size=2, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        batch.edge_feature)
        return batch


class GeneralConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralConv, self).__init__()
        self.model = GeneralConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class GeneralEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        edge_feature=batch.edge_feature)
        return batch


class GeneralSampleEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralSampleEdgeConv, self).__init__()
        self.model = GeneralEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        edge_mask = torch.rand(batch.edge_index.shape[1]) < cfg.gnn.keep_edge
        edge_index = batch.edge_index[:, edge_mask]
        edge_feature = batch.edge_feature[edge_mask, :]
        batch.node_feature = self.model(batch.node_feature, edge_index,
                                        edge_feature=edge_feature)
        return batch


class GraphRecurrentLayerWrapper(nn.Module):
    """
    The most general wrapper for graph recurrent layer, users can customize
        (1): the GNN block for message passing.
        (2): the update block takes {previous embedding, new node feature} and
            returns new node embedding.
    """
    def __init__(self, name, dim_in, dim_out, has_act=True, has_bn=True,
                 has_l2norm=False, id=0, **kwargs):
        super(GraphRecurrentLayerWrapper, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer_id = id
        has_bn = has_bn and cfg.gnn.batchnorm
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.layer = layer_dict[name](dim_in, dim_out,
                                      bias=not has_bn, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(
                dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if cfg.gnn.dropout > 0:
            layer_wrapper.append(nn.Dropout(
                p=cfg.gnn.dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[cfg.gnn.act])
        self.post_layer = nn.Sequential(*layer_wrapper)
        self.embedding_updater = construct_update_block(self.dim_in,
                                                        self.dim_out,
                                                        self.layer_id)

    def _init_hidden_state(self, batch):
        # Initialize hidden states of all nodes to zero.
        if not isinstance(batch.node_states[self.layer_id], torch.Tensor):
            batch.node_states[self.layer_id] = torch.zeros(
                batch.node_feature.shape[0], self.dim_out).to(
                batch.node_feature.device)

    def forward(self, batch):
        # Message passing.
        batch = self.layer(batch)
        batch.node_feature = self.post_layer(batch.node_feature)
        if self.has_l2norm:
            batch.node_feature = F.normalize(batch.node_feature, p=2, dim=1)

        self._init_hidden_state(batch)
        # Compute output from updater block.
        node_states_new = self.embedding_updater(batch)
        batch.node_states[self.layer_id] = node_states_new
        batch.node_feature = batch.node_states[self.layer_id]
        return batch


def construct_update_block(dim_in, dim_out, layer_id):
    # Helper function to construct an embedding updating module.
    # GRU-based models.
    # TODO: for code-release: make this clear.
    if cfg.gnn.embed_update_method in ['masked_gru', 'moving_average_gru', 'gru']:
        if cfg.gnn.gru_kernel == 'linear':
            # Simple GRU.
            return GRUUpdater(dim_in, dim_out, layer_id)
        else:
            # GNN-empowered GRU.
            return GraphConvGRUUpdater(dim_in, dim_out, layer_id,
                                       layer_dict[cfg.gnn.gru_kernel])
    elif cfg.gnn.embed_update_method == 'mlp':
        return MLPUpdater(dim_in, dim_out, layer_id, cfg.gnn.mlp_update_layers)
    else:
        raise NameError(f'Unknown embedding update method: {cfg.gnn.embed_update_method}.')


class MovingAverageUpdater(nn.Module):
    # TODO: complete this for code release.
    def __init__(self,):
        raise NotImplementedError()

    def forward(self, batch):
        # TODO: copy from the old implementation.
        raise NotImplementedError()


class MLPUpdater(nn.Module):
    """
    Node embedding update block using simple MLP.
    embedding_new <- MLP([embedding_old, node_feature_new])
    """
    def __init__(self, dim_in, dim_out, layer_id, num_layers):
        super(MLPUpdater, self).__init__()
        self.layer_id = layer_id
        # FIXME:
        assert num_layers > 1, 'There is a problem with layer=1 now, pending fix.'
        self.mlp = MLP(dim_in=dim_in + dim_out, dim_out=dim_out,
                       num_layers=num_layers)

    def forward(self, batch):
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        concat = torch.cat((H_prev, X), axis=1)
        H_new = self.mlp(concat)
        batch.node_states[self.layer_id] = H_new
        return H_new


class GRUUpdater(nn.Module):
    """
    Node embedding update block using standard GRU and variations of it.
    """
    def __init__(self, dim_in, dim_out, layer_id):
        # dim_in (dim of X): dimension of input node_feature.
        # dim_out (dim of H): dimension of previous and current hidden states.
        # forward(X, H) --> H.
        super(GRUUpdater, self).__init__()
        self.layer_id = layer_id
        self.GRU_Z = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # reset gate.
        self.GRU_R = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Sigmoid())
        # new embedding gate.
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(dim_in + dim_out, dim_out, bias=True),
            nn.Tanh())
    
    def forward(self, batch):
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        Z = self.GRU_Z(torch.cat([X, H_prev], dim=1))
        R = self.GRU_R(torch.cat([X, H_prev], dim=1))
        H_tilde = self.GRU_H_Tilde(torch.cat([X, R * H_prev], dim=1))
        H_gru = Z * H_prev + (1 - Z) * H_tilde

        if cfg.gnn.embed_update_method == 'masked_gru':
            # Update for active nodes only, use output from GRU.
            keep_mask = (batch.node_degree_new == 0)
            H_out = H_gru
            # Reset inactive nodes' embedding.
            H_out[keep_mask, :] = H_prev[keep_mask, :]
        elif cfg.gnn.embed_update_method == 'moving_average_gru':
            # Only update for active nodes, using moving average with output from GRU.
            H_out = H_prev * batch.keep_ratio + H_gru * (1 - batch.keep_ratio)
        elif cfg.gnn.embed_update_method == 'gru':
            # Update all nodes' embedding using output from GRU.
            H_out = H_gru
        return H_out


class GraphConvGRUUpdater(nn.Module):
    """
    Node embedding update block using GRU with internal GNN and variations of
    it.
    """
    def __init__(self, dim_in, dim_out, layer_id, conv):
        # dim_in (dim of X): dimension of input node_feature.
        # dim_out (dim of H): dimension of previous and current hidden states.
        # forward(X, H) --> H.
        super(GraphConvGRUUpdater, self).__init__()
        self.layer_id = layer_id
        
        self.GRU_Z = conv(dim_in=dim_in + dim_out, dim_out=dim_out)
        # reset gate.
        self.GRU_R = conv(dim_in=dim_in + dim_out, dim_out=dim_out)
        # new embedding gate.
        self.GRU_H_Tilde = conv(dim_in=dim_in + dim_out, dim_out=dim_out)

    def forward(self, batch):
        H_prev = batch.node_states[self.layer_id]
        X = batch.node_feature
        # Combe previous node embedding and current feature for message passing.
        batch_z = deepsnap.graph.Graph()
        batch_z.node_feature = torch.cat([X, H_prev], dim=1).clone()
        batch_z.edge_feature = batch.edge_feature.clone()
        batch_z.edge_index = batch.edge_index.clone()

        batch_r = deepsnap.graph.Graph()
        batch_r.node_feature = torch.cat([X, H_prev], dim=1).clone()
        batch_r.edge_feature = batch.edge_feature.clone()
        batch_r.edge_index = batch.edge_index.clone()

        # (num_nodes, dim_out)
        Z = nn.functional.sigmoid(self.GRU_Z(batch_z).node_feature)
        # (num_nodes, dim_out)
        R = nn.functional.sigmoid(self.GRU_R(batch_r).node_feature)

        batch_h = deepsnap.graph.Graph()
        batch_h.node_feature = torch.cat([X, R * H_prev], dim=1).clone()
        batch_h.edge_feature = batch.edge_feature.clone()
        batch_h.edge_index = batch.edge_index.clone()

        # (num_nodes, dim_out)
        H_tilde = nn.functional.tanh(self.GRU_H_Tilde(batch_h).node_feature)
        H_gru = Z * H_prev + (1 - Z) * H_tilde

        if cfg.gnn.embed_update_method == 'masked_gru':
            # Update for active nodes only, use output from GRU.
            keep_mask = (batch.node_degree_new == 0)
            H_out = H_gru
            # Reset inactive nodes' embedding.
            H_out[keep_mask, :] = H_prev[keep_mask, :]
        elif cfg.gnn.embed_update_method == 'moving_average_gru':
            # Only update for active nodes, using moving average with output from GRU.
            H_out = H_prev * batch.keep_ratio + H_gru * (1 - batch.keep_ratio)
        elif cfg.gnn.embed_update_method == 'gru':
            # Update all nodes' embedding using output from GRU.
            H_out = H_gru

        return H_out


layer_dict = {
    'linear': Linear,
    'mlp': MLP,
    'gcnconv': GCNConv,
    'sageconv': SAGEConv,
    'gatconv': GATConv,
    'splineconv': SplineConv,
    'ginconv': GINConv,
    'generalconv': GeneralConv,
    'generaledgeconv': GeneralEdgeConv,
    'generalsampleedgeconv': GeneralSampleEdgeConv,
}

# register additional convs
layer_dict = {**register.layer_dict, **layer_dict}
