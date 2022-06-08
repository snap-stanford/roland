from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.inits import glorot, zeros
from graphgym.config import cfg

from graphgym.register import register_layer

import pdb


class GeneralConvLayer(MessagePassing):
    r"""General graph convolution layer.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        """
        Args:
            in_channels: dimension of input node features.
            out_channels: dimension of output node embeddings.
            improved:
            cached:
            bias:
            **kwargs:
        """
        super(GeneralConvLayer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        r"""

        Args:
            edge_index: shape [2, num_edges]
            num_nodes:
            edge_weight:
            improved:
            dtype:

        Returns:

        """
        if edge_weight is None:
            # The unweighted case, edge_weight = 1.
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)
        # Add self-loops for nodes v such that (v, v) not in E, self-loops
        # have weights 1 or 2.
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index  # source node indices, destination node indices.
        # deg[v] = sum(edge_weight[i] for i in {0,1,...,num_nodes-1} s.t.
        # row[i] == v)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # normalize weight weight, w[u, v] = w[u, v] / sqrt(deg(u) * deg(v)).
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None):
        # Note: bias, if requested, will be applied after message aggregation.
        x = torch.matmul(x, self.weight)

        # If caching is requested and there exists previous cached edge_index.
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        # If caching is not requested or we need to initialize cache.
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                # Use the un-normalized edge weight.
                norm = edge_weight
            # Save (initialize) edge_index and normalized edge weights to cache.
            self.cached_result = edge_index, norm

        # Load from current cache.
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm,
                              edge_feature=edge_feature)

    def message(
            self,
            x_j: torch.Tensor,
            norm: Optional[torch.Tensor],
            edge_feature: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""
        Args:
            x_j: shape [num_edges, num_node_features]
            norm: shape [num_edges]
            edge_feature: [num_edges, num_edge_features]

        Returns:

        """
        if edge_feature is None:
            # If no additional edge features are provided, the message is simply
            # the weighted features of the source node j.
            return norm.view(-1, 1) * x_j if norm is not None else x_j
        else:
            # If there are edge features, add to node features before
            # applying edge weight.
            return norm.view(-1, 1) * (
                    x_j + edge_feature) if norm is not None else (
                    x_j + edge_feature)

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GeneralEdgeHeteConvLayer(MessagePassing):
    r"""General GNN layer, with arbitrary edge features.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(GeneralEdgeHeteConvLayer, self).__init__(aggr=cfg.gnn.agg,
                                                       **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.msg_direction = cfg.gnn.msg_direction

        # (1) Node transformation based on node type
        # todo: define node/edge type when constructing layers
        num_type = 2
        self.linear_node = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(num_type)])
        # todo: define node/edge type when constructing layers
        num_type = 3
        if self.msg_direction == 'single':
            # Edge messages are constructed using from features of the
            # source
            # node and the edge.
            # We do not need bias for this linear layer,
            # the bias, if requested, will be added after message
            # aggregation.
            self.linear_msg = nn.ModuleList([
                nn.Linear(out_channels + cfg.dataset.edge_dim, out_channels,
                          bias=False)
                for _ in range(num_type)])
        elif self.msg_direction == 'both':
            # Edge messages are constructed using features of both source
            # and
            # destination nodes and the edge.
            self.linear_msg = nn.ModuleList([
                nn.Linear(out_channels * 2 + cfg.dataset.edge_dim, out_channels,
                          bias=False)
                for _ in range(num_type)])
        else:
            raise ValueError(
                f'Unsupported message passing direction: '
                f'{self.msg_direction}.')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, batch, edge_weight=None):
        x = batch.node_feature
        edge_index = batch.edge_index
        for i,type in enumerate(batch.list_n_type):
            id = batch.node_type == i
            x_type = x[id, :]
            x_type = self.linear_node[i](x_type)
            if i == 0:
                x_out = torch.zeros(x.shape[0], x_type.shape[1],
                                    device=x.device)
            x_out.index_add_(0, id.nonzero().squeeze(), x_type)
        x = x_out

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm,
                              edge_feature=batch.edge_feature,
                              edge_type=batch.edge_type,
                              list_e_type=batch.list_e_type)

    def message(self, x_i, x_j, norm, edge_feature, edge_type, list_e_type):
        if self.msg_direction == 'both':
            x_j = torch.cat((x_i, x_j, edge_feature), dim=-1)
        elif self.msg_direction == 'single':
            x_j = torch.cat((x_j, edge_feature), dim=-1)
        else:
            raise ValueError(
                f'Unsupported message passing direction: {self.msg_direction}.')

        for i,type in enumerate(list_e_type):
            id = edge_type == i
            x_type = x_j[id, :]
            x_type = self.linear_msg[i](x_type)
            if i == 0:
                x_out = torch.zeros(x_j.shape[0], x_type.shape[1],
                                    device=x_j.device)
            x_out.index_add_(0, id.nonzero().squeeze(), x_type)
        return norm.view(-1, 1) * x_out if norm is not None else x_out

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GeneralEdgeHeteConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralEdgeHeteConv, self).__init__()
        self.model = GeneralEdgeHeteConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch)
        return batch


register_layer('generaledgeheteconv', GeneralEdgeHeteConv)
