import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros

from graphgym.register import register_layer


class GConvLSTMBaseline(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int = 7,
                 normalization: str = "sym", id: int = -1, bias: bool = True):
        super(GConvLSTMBaseline, self).__init__()
        assert id >= 0, 'kwarg id is required.'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()
        self.id = id

    def _create_parameters_and_layers(self):
        # Initial feature extraction.
        self.feature_gnn = ChebConv(in_channels=self.out_channels,
                                    out_channels=self.out_channels,
                                    K=self.K,
                                    normalization=self.normalization,
                                    bias=self.bias)
        # Input gate.
        self.w_x_i = nn.Linear(self.in_channels, self.out_channels,
                               bias=False)
        self.w_h_i = nn.Linear(self.out_channels, self.out_channels,
                               bias=False)
        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))
        # Forget gate.
        self.w_x_f = nn.Linear(self.in_channels, self.out_channels,
                               bias=False)
        self.w_h_f = nn.Linear(self.out_channels, self.out_channels,
                               bias=False)
        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))
        # Cell state.
        self.w_x_c = nn.Linear(self.in_channels, self.out_channels,
                               bias=False)
        self.w_h_c = nn.Linear(self.out_channels, self.out_channels,
                               bias=False)
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))
        # Output gate.
        self.w_x_o = nn.Linear(self.in_channels, self.out_channels,
                               bias=False)
        self.w_h_o = nn.Linear(self.out_channels, self.out_channels,
                               bias=False)

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if not isinstance(H, torch.Tensor):
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if not isinstance(C, torch.Tensor):
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                 edge_weight: torch.FloatTensor = None,
                 H: torch.FloatTensor = None,
                 C: torch.FloatTensor = None
                 ) -> (torch.FloatTensor, torch.FloatTensor):
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        # node feature from GNN.
        X = self.feature_gnn(X, edge_index, edge_weight)
        # input gate.
        I = self.w_x_i(X) + self.w_h_i(H) + (self.w_c_i * C) + self.b_i
        I = torch.sigmoid(I)
        # forget gate.
        F = self.w_x_f(X) + self.w_h_f(H) + (self.w_c_f * C) + self.b_f
        F = torch.sigmoid(F)
        # cell state.
        T = self.w_x_c(X) + self.w_h_c(H) + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        # output gate.
        O = self.w_x_o(X) + self.w_h_o(H) + (self.w_c_o * C) + self.b_o
        O = torch.sigmoid(O)
        # new hidden state.
        H = O * torch.tanh(C)
        return H, C

    def forward(self, batch):
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        else:
            edge_weight = None

        H, C = self._forward(X=batch.node_feature,
                             edge_index=batch.edge_index,
                             edge_weight=edge_weight,
                             H=batch.node_states[self.id],
                             C=batch.node_cells[self.id])

        batch.node_states[self.id] = H
        batch.node_cells[self.id] = C
        batch.node_feature = H
        return batch


register_layer('gconv_lstm_baseline', GConvLSTMBaseline)
