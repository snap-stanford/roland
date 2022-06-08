import torch
from torch_geometric.nn import GCNConv

from graphgym.register import register_layer


class TGCN(torch.nn.Module):
    r"""An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is True.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 id: int = -1):
        super(TGCN, self).__init__()
        assert id >= 0, 'kwarg id is required.'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.id = id

        self.graph_conv1 = GCNConv(in_channels=self.in_channels + self.out_channels,
                                   out_channels=self.out_channels * 2,
                                   improved=self.improved,
                                   cached=self.cached,
                                   normalize=True,
                                   bias=True,
                                   add_self_loops=True)
        # NOTE: possible issues here, by forcefully setting parameters.
        # but the original TGCN implementation initialized bias to ones.
        self.graph_conv1.bias.data = torch.ones_like(self.graph_conv1.bias.data)

        self.graph_conv2 = GCNConv(in_channels=self.in_channels + self.out_channels,
                                   out_channels=self.out_channels,
                                   improved=self.improved,
                                   cached=self.cached,
                                   normalize=True,
                                   bias=True,
                                   add_self_loops=True)

        # self._create_parameters_and_layers()
    #
    # def _create_update_gate_parameters_and_layers(self):
    #     self.conv_z = GCNConv(in_channels=self.in_channels,
    #                           out_channels=self.out_channels,
    #                           improved=self.improved,
    #                           cached=self.cached,
    #                           normalize=True,
    #                           bias=True,
    #                           add_self_loops=True)
    #
    #     self.linear_z = torch.nn.Linear(2 * self.out_channels,
    #                                     self.out_channels)
    #
    # def _create_reset_gate_parameters_and_layers(self):
    #     self.conv_r = GCNConv(in_channels=self.in_channels,
    #                           out_channels=self.out_channels,
    #                           improved=self.improved,
    #                           cached=self.cached,
    #                           normalize=True,
    #                           bias=True,
    #                           add_self_loops=True)
    #
    #     self.linear_r = torch.nn.Linear(2 * self.out_channels,
    #                                     self.out_channels)
    #
    # def _create_candidate_state_parameters_and_layers(self):
    #     self.conv_h = GCNConv(in_channels=self.in_channels,
    #                           out_channels=self.out_channels,
    #                           improved=self.improved,
    #                           cached=self.cached,
    #                           normalize=True,
    #                           add_self_loops=True)
    #
    #     self.linear_h = torch.nn.Linear(2 * self.out_channels,
    #                                     self.out_channels)
    #
    # def _create_parameters_and_layers(self):
    #     self._create_update_gate_parameters_and_layers()
    #     self._create_reset_gate_parameters_and_layers()
    #     self._create_candidate_state_parameters_and_layers()
    #
    # def _set_hidden_state(self, X, H):
    #     if not isinstance(H, torch.Tensor):
    #         H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
    #     return H

    # def _calculate_update_gate(self, X, edge_index, edge_weight, H):
    #     Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)
    #     Z = self.linear_z(Z)
    #     Z = torch.sigmoid(Z)
    #     return Z
    #
    # def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
    #     R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
    #     R = self.linear_r(R)
    #     R = torch.sigmoid(R)
    #     return R
    #
    # def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
    #     H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R],
    #                         axis=1)
    #     H_tilde = self.linear_h(H_tilde)
    #     H_tilde = torch.tanh(H_tilde)
    #     return H_tilde
    #
    # def _calculate_hidden_state(self, Z, H, H_tilde):
    #     H = Z * H + (1 - Z) * H_tilde
    #     return H

    def _forward(self, X: torch.FloatTensor, edge_index: torch.LongTensor,
                 edge_weight: torch.FloatTensor = None,
                 H: torch.FloatTensor = None) -> torch.FloatTensor:
        # breakpoint()
        if not isinstance(H, torch.Tensor):
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        # print('H:', H.shape)
        concatenation = torch.sigmoid(
            self.graph_conv1(torch.cat([X, H], dim=1), edge_index, edge_weight)
        )
        # print('concatenation:', concatenation.shape)
        # r = concatenation[:, :self.out_channels]
        # u = concatenation[:, self.out_channels:]
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # print('r:', r.shape)
        # print('u:', u.shape)
        c = torch.tanh(self.graph_conv2(
            torch.cat([X, H * r], dim=1), edge_index, edge_weight
        ))
        # print('c:', c.shape)
        H = u * H + (1.0 - u) * c
        # breakpoint()
        return H

    def forward(self, batch):
        if hasattr(batch, 'edge_weight'):
            edge_weight = batch.edge_weight
        else:
            edge_weight = None

        H = self._forward(X=batch.node_feature, edge_index=batch.edge_index,
                          edge_weight=edge_weight,
                          H=batch.node_states[self.id])
        batch.node_states[self.id] = H
        batch.node_feature = H
        return batch


register_layer('tgcn', TGCN)
