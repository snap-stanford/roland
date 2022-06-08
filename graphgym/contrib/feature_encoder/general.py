"""
Basic edge encoder for temporal graphs, this encoder does not assume edge dim,
this encoder uses linear layers to contract/expand raw edge features to
dimension cfg.transaction.feature_amount_dim + feature_time_dim for consistency.
"""

import deepsnap
import torch
import torch.nn as nn
from graphgym.config import cfg
from graphgym.register import register_edge_encoder


class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim: int):
        # emb_dim is not used here.
        super(LinearEdgeEncoder, self).__init__()
        # For consistency, the edge features will be map to this dimension
        # on the BSI dataset.
        expected_dim = cfg.transaction.feature_amount_dim \
            + cfg.transaction.feature_time_dim
        
        self.linear = nn.Linear(cfg.dataset.edge_dim, expected_dim)
        cfg.dataset.edge_dim = expected_dim

    def forward(self, batch: deepsnap.batch.Batch) -> deepsnap.batch.Batch:
        batch.edge_feature = self.linear(batch.edge_feature)
        return batch


register_edge_encoder('roland_general', LinearEdgeEncoder)
