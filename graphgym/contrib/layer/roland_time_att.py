"""
The online message passing layer with attention over time.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import softmax, add_remaining_self_loops
from torch_scatter import scatter_add

from graphgym.config import cfg
from graphgym.register import register_layer


class TimeEdgeAttConvLayer(MessagePassing):
    r"""A graph convolution layer with attention over edges based on
        transaction times associated with edges.
    """
    def __init__(self, in_channels, out_channels, task_channels=None,
                 improved=False, cached=False, bias=True, **kwargs):
        """
        Args:
            in_channels: node embedding dimension = batch.node_feature.shape[1]
            out_channels: new node embedding dimension.

        NOTE: the cached and improved functions are not implemented.
        """
        super(TimeEdgeAttConvLayer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        # Original configurations from the general edge conv layer.
        self.heads = cfg.gnn.att_heads
        self.in_channels = int(in_channels // self.heads * self.heads)
        self.out_channels = int(out_channels // self.heads * self.heads)
        self.task_channels = task_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.msg_direction = cfg.gnn.msg_direction
        self.negative_slope = 0.2

        self.head_channels = out_channels // self.heads
        self.scaling = self.head_channels ** -0.5

        if self.msg_direction == 'single':
            # Edge messages are constructed using from features of the source
            # node and the edge.
            self.linear_msg = nn.Linear(in_channels + cfg.dataset.edge_dim,
                                        out_channels, bias=False)
        elif self.msg_directioan == 'both':
            # Edge messages are constructed using features of both source and
            # destination nodes and the edge.
            self.linear_msg = nn.Linear(in_channels * 2 + cfg.dataset.edge_dim,
                                        out_channels, bias=False)
        else:
            raise ValueError(
                f'Unsupported message passing direction: {self.msg_direction}.')

        if self.task_channels is not None:
            self.att_task = Parameter(
                torch.Tensor(1, self.heads, self.task_channels))

        if cfg.gnn.att_final_linear:
            self.linear_final = nn.Linear(out_channels, out_channels,
                                          bias=False)
        if cfg.gnn.att_final_linear_bn:
            self.linear_final_bn = nn.BatchNorm1d(out_channels, eps=cfg.bn.eps,
                                                  momentum=cfg.bn.mom)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # A list of days used to construct positional encoding.
        self.pos_enc_periods = cfg.transaction.time_enc_dim

        if not isinstance(self.pos_enc_periods, list):
            raise TypeError(
                f'pos_enc_dim(time_enc_dim) must be an a list of days.')
        # the attention layer maps periods captured previously
        # to attention heads. 2*len() b/c sin and cos for 1 period.
        # (2*len(pos_enc_periods) --> heads).
        self.attention_layer = nn.Linear(2*len(self.pos_enc_periods),
                                         self.heads, bias=False)
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

    def delta_time_encoding(self, delta_t: torch.Tensor) -> torch.Tensor:
        r"""Construct the encoding for time delta.
        Designed for the enc(diff(t1, t2)) pipeline.

        This time enc is supposed to capture various periodicity using
            sin and cos functions.
        For a given delta_t, the corresponding encoding tensor looks like
        pos_enc = (
            sin(omega_1 * delta_t)
            cos(omega_1 * delta_t)
            sin(omega_2 * delta_t)
            cos(omega_2 * delta_t)
            ...
            sin(omega_d * delta_t)
            cos(omega_d * delta_t)
        )
        for omega_i in self.pos_enc_periods.
        where omegas are frequencies = 2*pi/periods.

        Args:
            delta_t: a tensor of shape (num_edges,) in which delta_t[k]
                indicates forecast_time - transaction time of edge k,
                measured in number of seconds to comply with timestamp.

        Returns:
            a tensor with shape (num_edges, 2*len(self.pos_enc_dim)).
        """
        # convert into number of days, 86400 sec in 1 day.
        delta_t_norm = delta_t / torch.scalar_tensor(86400)
        delta_t_norm = delta_t_norm.view(-1, 1)  # (num_edges, 1)

        enc_list = list()
        for period in self.pos_enc_periods:
            omega = 2 * np.pi / period
            enc_list.append(torch.sin(omega * delta_t_norm))
            enc_list.append(torch.cos(omega * delta_t_norm))
        pos_enc = torch.cat(enc_list, dim=1)
        # (E, encoding_dim = 2*len(self.pos_enc_periods))
        assert pos_enc.shape == (delta_t.shape[0], 2*len(self.pos_enc_periods))
        return pos_enc

    def forward(self, x, edge_index, edge_feature, edge_time_delta,
                edge_weight=None):
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

        return self.propagate(edge_index,
                              x=x,
                              norm=norm,
                              edge_feature=edge_feature,
                              delta_t=edge_time_delta)

    def message(self, edge_index_i, x_i, x_j, norm, size_i, edge_feature,
                delta_t):
        """
        Computes message from node j to node i, with time encoding.
        Returns:
            edgewise message with shape (E, heads, head_channels).
            this message will be reshaped in update().
        """
        if self.msg_direction == 'both':
            x_j = torch.cat((x_i, x_j, edge_feature), dim=-1)
        else:
            x_j = torch.cat((x_j, edge_feature), dim=-1)
        # (E, msg_dim)
        x_j = self.linear_msg(x_j)  # (E, out_channels).
        # heads * head_channels = out_channels.
        x_j = x_j.view(-1, self.heads, self.head_channels)

        # Construct attention based on time encoding.
        # alpha should be (E, heads, 1)
        enc_t = self.delta_time_encoding(delta_t)  # (E, enc_dim)
        alpha = self.attention_layer(enc_t)  # (E, heads)
        alpha = F.leaky_relu(alpha, self.negative_slope)  # (E, heads)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        # scatter softmax (E, heads)
        alpha = alpha.view(-1, self.heads, 1)
        return norm.view(-1,
                         1) * x_j * alpha if norm is not None else x_j * alpha

    def update(self, aggr_out):
        """(E, heads, head_channels) --> (E, out_channels)"""
        aggr_out = aggr_out.view(-1, self.out_channels)
        if cfg.gnn.att_final_linear_bn:
            aggr_out = self.linear_final_bn(aggr_out)
        if cfg.gnn.att_final_linear:
            aggr_out = self.linear_final(aggr_out)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}) with attention periods: 2pi*{} (days)'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.att_periods)


class TimeEdgeAttConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(TimeEdgeAttConv, self).__init__()
        self.model = TimeEdgeAttConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(x=batch.node_feature,
                                        edge_index=batch.edge_index,
                                        edge_feature=batch.edge_feature,
                                        edge_time_delta=batch.edge_time_delta)
        return batch


register_layer('att_over_time_edge_conv', TimeEdgeAttConv)
