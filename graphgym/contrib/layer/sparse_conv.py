import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from torch_sparse import SparseTensor, matmul

from graphgym.config import cfg
from graphgym.register import register_layer


class SparseEdgeConvLayer(MessagePassing):
    r"""General GNN layer, with arbitrary edge features.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(SparseEdgeConvLayer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.msg_direction = cfg.gnn.msg_direction

        self.linear_edge = nn.Sequential(
            nn.Linear(cfg.dataset.edge_dim, 1),
            nn.Sigmoid())
        self.linear_node = nn.Linear(in_channels, out_channels)

        # if self.msg_direction == 'single':
        #     # Edge messages are constructed using from features of the source node and the edge.
        #     # We do not need bias for this linear layer,
        #     # the bias, if requested, will be added after message aggregation.
        #     self.linear_msg = nn.Linear(in_channels + cfg.dataset.edge_dim,
        #                                 out_channels, bias=False)
        # elif self.msg_direction == 'both':
        #     # Edge messages are constructed using features of both source and destination nodes and the edge.
        #     self.linear_msg = nn.Linear(in_channels * 2 + cfg.dataset.edge_dim,
        #                                 out_channels, bias=False)
        # else:
        #     raise ValueError(f'Unsupported message passing direction: {self.msg_direction}.')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_feature):
        # breakpoint()
        edge_feature = self.linear_edge(edge_feature)  # hetero-able here.
        # effectively an attention mechanism.
        x = self.linear_node(x)  # hetero-able here.
        num_nodes = x.shape[0]
        W = SparseTensor(row=edge_index[0], col=edge_index[1],
                         value=edge_feature.squeeze(),
                         sparse_sizes=(num_nodes, num_nodes))
        out = self.propagate(edge_index=W, x=x)
        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        # return adj_t @ x
        return matmul(adj_t, x, reduce='mean')

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SparseEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SparseEdgeConv, self).__init__()
        self.model = SparseEdgeConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature,
                                        batch.edge_index,
                                        batch.edge_feature)
        return batch


register_layer('sparse_edge_conv', SparseEdgeConv)
