"""
An implementation of the Evolving Graph Convolutional Hidden Layer.
For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_
"""
import torch
import torch.nn as nn
from torch.nn import LSTM, GRU
from torch_geometric.nn import GCNConv
import math

from graphgym.register import register_layer


class EvolveGCNO(torch.nn.Module):
    """
    The O-version of Evolve GCN, the H-version is too restricted, and the
    transaction graph is more about constructing meaningful embeddings from
    graph structure, initial node features are not that important.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False,
                 cached: bool = False, normalize: bool = True,
                 add_self_loops: bool = True,
                 bias: bool = False,
                 id: int = -1):
        """
        NOTE: EvolveGCNO does not change size of representation,
            i.e., out_channels == in_channels.
        This can be easily modified, but not necessary in the ROLAND use case
            as we have out_channels == in_channels == inner_dim.
        """
        super(EvolveGCNO, self).__init__()
        assert id >= 0, 'kwarg id is required.'

        self.in_channels = in_channels
        assert in_channels == out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.id = id
        self._create_layers()
        std = 1. / math.sqrt(in_channels)
        self.conv_layer.lin.weight.data.uniform_(-std, std)

    def _create_layers(self):
        # self.recurrent_layer = GRU(input_size=self.in_channels,
        #                            hidden_size=self.in_channels,
        #                            num_layers=1)

        # self.update_gate = nn.Sequential(
        #     nn.Linear(2 * self.in_channels, self.in_channels, bias=True),
        #     nn.Sigmoid()
        # )

        # self.reset_gate = nn.Sequential(
        #     nn.Linear(2 * self.in_channels, self.in_channels, bias=True),
        #     nn.Sigmoid()
        # )

        # self.h_tilde = nn.Sequential(
        #     nn.Linear(2 * self.in_channels, self.in_channels),
        #     nn.Tanh()
        # )

        self.update = mat_GRU_gate(self.in_channels,
                                   self.in_channels,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(self.in_channels,
                                  self.in_channels,
                                  torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(self.in_channels,
                                   self.in_channels,
                                   torch.nn.Tanh())

        self.conv_layer = GCNConv(in_channels=self.in_channels,
                                  out_channels=self.in_channels,
                                  improved=self.improved,
                                  cached=self.cached,
                                  normalize=self.normalize,
                                  add_self_loops=self.add_self_loops,
                                  bias=True)

    def _forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                 edge_weight: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        # W = self.conv_layer.lin.weight[None, :, :].detach().clone()
        # # W has shape (1, inner_dim, inner_dim),
        # # corresponds to (seq_len, batch, input_size).
        # W, _ = self.recurrent_layer(W, W.clone())
        # self.conv_layer.lin.weight = torch.nn.Parameter(W.squeeze())
        # # breakpoint()
        W = self.conv_layer.lin.weight.detach().clone()
        # update = self.update_gate(torch.cat((W, W), axis=1))
        # reset = self.reset_gate(torch.cat((W, W), axis=1))
        # h_tilde = self.h_tilde(torch.cat((W, reset * W), axis=1))
        # W = (1 - update) * W + update * h_tilde

        update = self.update(W, W)
        reset = self.reset(W, W)

        h_cap = reset * W
        h_cap = self.htilda(W, h_cap)

        new_W = (1 - update) * W + update * h_cap

        self.conv_layer.lin.weight.data = new_W.clone()
        X = self.conv_layer(X, edge_index, edge_weight)
        return X

    def forward(self, batch):
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        else:
            edge_weight = None
        out = self._forward(batch.node_feature, batch.edge_index, edge_weight)
        # For consistency with the training pipeline, node_states are not
        # used in this model.
        batch.node_states[self.id] = out
        batch.node_feature = out
        return batch


register_layer('evolve_gcn_o', EvolveGCNO)


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = nn.Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = nn.Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = nn.Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) +
                              self.U.matmul(hidden) +
                              self.bias)

        return out
