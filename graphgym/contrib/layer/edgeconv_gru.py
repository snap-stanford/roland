"""
A general version of gated recurrent unit with internal graph convolution.
"""
import torch
import torch.nn as nn

from graphgym.contrib.layer.residual_edge_conv import ResidualEdgeConvLayer
from graphgym.register import register_layer


class GeneralEdgeConvGRU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, id: int = -1,
                 bias: bool = True):
        super(GeneralEdgeConvGRU, self).__init__()
        assert id >= 0, 'kwarg id is required.'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self._create_parameters_and_layers()
        self.id = id

    def _create_update_gate_parameters_and_layers(self):
        self.conv_x_z = ResidualEdgeConvLayer(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              bias=self.bias)

        self.conv_h_z = ResidualEdgeConvLayer(in_channels=self.out_channels,
                                              out_channels=self.out_channels,
                                              bias=self.bias)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_x_r = ResidualEdgeConvLayer(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              bias=self.bias)

        self.conv_h_r = ResidualEdgeConvLayer(in_channels=self.out_channels,
                                              out_channels=self.out_channels,
                                              bias=self.bias)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_x_h = ResidualEdgeConvLayer(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              bias=self.bias)

        self.conv_h_h = ResidualEdgeConvLayer(in_channels=self.out_channels,
                                              out_channels=self.out_channels,
                                              bias=self.bias)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if not isinstance(H, torch.Tensor):
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_feature, H):
        Z = self.conv_x_z(X, edge_index, edge_feature=edge_feature)
        Z = Z + self.conv_h_z(H, edge_index, edge_feature=edge_feature)
        Z = torch.sigmoid(Z)  # (num_nodes, hidden_dim)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_feature, H):
        R = self.conv_x_r(X, edge_index, edge_feature=edge_feature)
        R = R + self.conv_h_r(H, edge_index, edge_feature=edge_feature)
        R = torch.sigmoid(R)  # (num_nodes, hidden_dim)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_feature, H, R):
        H_tilde = self.conv_x_h(X, edge_index, edge_feature=edge_feature)
        H_tilde = H_tilde + self.conv_h_h(H * R, edge_index,
                                          edge_feature=edge_feature)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde  # (num_nodes, hidden_dim)

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H  # (num_nodes, hidden_dim)

    def forward(self, batch):
        X = batch.node_feature
        edge_index = batch.edge_index
        edge_feature = batch.edge_feature
        H = batch.node_states[self.id]

        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_feature, H)
        R = self._calculate_reset_gate(X, edge_index, edge_feature, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_feature,
                                                  H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)

        batch.node_states[self.id] = H
        batch.node_feature = H
        return batch


register_layer('edge_conv_gru', GeneralEdgeConvGRU)
