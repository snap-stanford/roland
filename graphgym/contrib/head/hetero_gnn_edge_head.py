import torch
import torch.nn as nn

from graphgym.config import cfg
from graphgym.models.layer import MLP
from graphgym.register import register_head


class HeteroEdgeHead(nn.Module):
    r"""The GNN head module for edge prediction tasks. This module takes a (batch of) graphs and
    outputs ...
    """

    def __init__(self, dim_in: int, dim_out: int):
        """ Head of Edge and link prediction models.
        Args:
            dim_out: output dimension. For binary prediction, dim_out=1.
        """
        # Use dim_in for graph conv, since link prediction dim_out could be
        # binary
        # E.g. if decoder='dot', link probability is dot product between
        # node embeddings, of dimension dim_in
        super(HeteroEdgeHead, self).__init__()
        # module to decode edges from node embeddings

        if cfg.model.edge_decoding == 'concat':
            # Only use node features.
            self.layer_post_mp = MLP(dim_in * 2, dim_out,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            # requires parameter
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=-1))
        elif cfg.model.edge_decoding == 'edgeconcat':
            # Use both node and edge features.
            self.layer_post_mp = MLP(dim_in * 2 + cfg.dataset.edge_dim, dim_out,
                                     num_layers=cfg.gnn.layers_post_mp,
                                     bias=True)
            # requires parameter
            self.decode_module = lambda v1, v2, edge: \
                self.layer_post_mp(torch.cat((v1, v2, edge), dim=-1))
        else:
            raise NotImplementedError
        # else:
        #     if dim_out > 1:
        #         raise ValueError(
        #             'Binary edge decoding ({})is used for multi-class '
        #             'edge/link prediction.'.format(cfg.model.edge_decoding))
        #     self.layer_post_mp = MLP(dim_in, dim_in,
        #                              num_layers=cfg.gnn.layers_post_mp,
        #                              bias=True)
        #     if cfg.model.edge_decoding == 'dot':
        #         self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
        #     elif cfg.model.edge_decoding == 'cosine_similarity':
        #         self.decode_module = nn.CosineSimilarity(dim=-1)
        #     else:
        #         raise ValueError('Unknown edge decoding {}.'.format(
        #             cfg.model.edge_decoding))

    # def _apply_index(self, batch):
    #     return batch.node_feature[batch.edge_label_index], \
    #            batch.edge_label

    def forward_pred(self, batch):
        r"""Makes predictions for each message type.
        Args:
            batch: HeteroGraph.
        Returns:
            prediction: pred_dict: Dict[MessageType, Predictions]
            ground_truth: batch.edge_label: Dict[MessageType, TrueLabels]
        """
        pred_dict = dict()  # Dict[MessageType, Predictions]
        for (s, r, d) in batch.edge_label_index.keys():
            edge_label_index = batch.edge_label_index[(s, r, d)]
            nodes_first = batch.node_feature[s][edge_label_index[0, :]]
            nodes_second = batch.node_feature[d][edge_label_index[1, :]]
            if cfg.model.edge_decoding == 'edgeconcat':
                raise NotImplementedError
                # TODO: still need to implement this.
                # edge_feature = torch.index_select(
                #     batch.edge_feature[(s, r, d)], 0,
                #     batch.edge_split_index)
                # pred = self.decode_module(nodes_first, nodes_second, edge_feature)
            else:
                # solely based on node embeddings.
                pred = self.decode_module(nodes_first, nodes_second)
            pred_dict[(s, r, d)] = pred
        return pred_dict, batch.edge_label

    def forward(self, batch):
        if cfg.model.edge_decoding != 'concat' and \
            cfg.model.edge_decoding != 'edgeconcat':
            batch = self.layer_post_mp(batch)
        pred, label = self.forward_pred(batch)
        return pred, label


register_head('hetero_edge_head', HeteroEdgeHead)
