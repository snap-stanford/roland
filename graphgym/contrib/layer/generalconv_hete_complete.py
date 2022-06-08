import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset

from graphgym.config import cfg
from graphgym.register import register_layer


class EdgeConvLayer(MessagePassing):
    r"""
    Args:
        in_channels_neigh (int): The input dimension of the end node type.
        out_channels (int): The dimension of the output.
        in_channels_self (int): The input dimension of the start node type.
            Default is `None` where the `in_channels_self` is equal to
            `in_channels_neigh`.
    """
    def __init__(self, in_channels_neigh, out_channels, in_channels_self=None):
        super(EdgeConvLayer, self).__init__(aggr=cfg.gnn.agg)
        self.in_channels_neigh = in_channels_neigh
        if in_channels_self is None:
            self.in_channels_self = in_channels_neigh
        else:
            self.in_channels_self = in_channels_self
        self.out_channels = out_channels
        self.edge_channels = cfg.dataset.edge_dim
        self.msg_direction = cfg.gnn.msg_direction

        self.lin_neigh = nn.Linear(self.in_channels_neigh, self.out_channels)
        self.lin_self = nn.Linear(self.in_channels_self, self.out_channels)

        if self.msg_direction == 'single':
            self.lin_update = nn.Linear(
                self.out_channels + cfg.dataset.edge_dim,
                self.out_channels)
        elif self.msg_direction == 'both':
            self.lin_update = nn.Linear(
                self.out_channels * 2 + cfg.dataset.edge_dim,
                self.out_channels)
        else:
            raise ValueError

    def forward(self, node_feature_neigh, node_feature_self, edge_index,
                edge_feature, edge_weight=None, size=None):
        return self.propagate(
            edge_index, size=size,
            node_feature_neigh=node_feature_neigh,
            node_feature_self=node_feature_self,
            edge_feature=edge_feature,
            edge_weight=edge_weight
        )

    def message(self, node_feature_neigh_j, node_feature_self_i,
                edge_feature, edge_weight):
        if self.msg_direction == 'single':
            node_feature_neigh_j = self.lin_neigh(node_feature_neigh_j)
            return torch.cat([node_feature_neigh_j, edge_feature], dim=-1)
        else:
            node_feature_neigh_j = self.lin_neigh(node_feature_neigh_j)
            node_feature_self_i = self.lin_self(node_feature_self_i)
            return torch.cat(
                [node_feature_neigh_j, edge_feature, node_feature_self_i],
                dim=-1)

    def update(self, aggr_out):
        aggr_out = self.lin_update(aggr_out)
        return aggr_out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(neigh: {self.in_channels_neigh}, self: {self.in_channels_self}, "
            f"edge: {self.edge_channels},"
            f"out: {self.out_channels})"
        )


class HeteroGNNWrapperConv(torch.nn.Module):
    def __init__(self, convs, dim_in, dim_out, aggr='add'):
        super(HeteroGNNWrapperConv, self).__init__()
        self.convs = convs
        # self.modules = torch.nn.ModuleList(convs.values())
        self.dim_in = dim_in
        self.dim_out = dim_out
        # NOTE: this aggregation is different from cfg.gnn.agg
        assert aggr in ['add', 'mean', 'max', None]
        self.aggr = aggr
        self.reset_parameters()  # TODO: something like this?

    def reset_parameters(self):
        for conv in self.convs.values():
            reset(conv)

    def forward(self, node_features, edge_indices, edge_features):
        r"""The forward function for `HeteroConv`.

        Args:
            node_features (dict): A dictionary each key is node type and the
                corresponding value is a node feature tensor.
            edge_indices (dict): A dictionary each key is message type and the
                corresponding value is an edge index tensor.
            edge_features (dict): A dictionary each key is edge type and the
                corresponding value is an edge feature tensor.
        """
        # node embedding computed from each message type
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            # neigh_type --(edge_type)--> self_type
            neigh_type, edge_type, self_type = message_key
            node_feature_neigh = node_features[neigh_type]
            node_feature_self = node_features[self_type]
            # edge_feature = edge_features[edge_type]
            edge_feature = edge_features[message_key]
            edge_index = edge_indices[message_key]

            message_type_emb[message_key] = (
                self.convs[str(message_key)](
                    node_feature_neigh,
                    node_feature_self,
                    edge_index,
                    edge_feature
                )
            )

        # TODO: What if a type does not receive anything within the period?
        node_emb = {typ: [] for typ in node_features.keys()}

        for (_, _, tail), item in message_type_emb.items():
            node_emb[tail].append(item)

        # Aggregate multiple embeddings with the same tail.
        for node_type, embs in node_emb.items():
            if len(embs) == 0:
                # This type of nodes did not receive any incoming edge,
                # put all zeros, keep_ratio will be 1, this does not matter.
                node_emb[node_type] = torch.zeros((
                    node_features[node_type].shape[0],
                    self.dim_out
                )).to(cfg.device)
            elif len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb  # Dict[NodeType, NodeEmb]

    def aggregate(self, xs):
        x = torch.stack(xs, dim=-1)
        if self.aggr == "add":
            return x.sum(dim=-1)
        elif self.aggr == "mean":
            return x.mean(dim=-1)
        elif self.aggr == "max":
            return x.max(dim=-1)[0]


class HeteroGeneralEdgeConv(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(HeteroGeneralEdgeConv, self).__init__()
        convs = nn.ModuleDict()
        for s, r, d in cfg.dataset.message_types:
            module = EdgeConvLayer(dim_in, dim_out)
            convs[str((s, r, d))] = module
        self.model = HeteroGNNWrapperConv(convs, dim_in, dim_out, 'mean')

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature,
                                        batch.edge_index,
                                        batch.edge_feature)
        return batch


register_layer('generaledgeheteconv_complete', HeteroGeneralEdgeConv)
