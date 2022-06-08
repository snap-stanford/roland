import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from graphgym.config import cfg
from graphgym.contrib.train import train_utils
from graphgym.utils.stats import node_degree


def get_edge_label(dataset, current, horizon, mode):
    if mode == 'before':
        edge_label = torch.cat([dataset[current + i].edge_label
                                for i in range(1, horizon + 1)], dim=0)
        edge_label_index = torch.cat([dataset[current + i].edge_label_index
                                      for i in range(1, horizon + 1)], dim=1)
    elif mode == 'at':
        edge_label = dataset[current + horizon].edge_label
        edge_label_index = dataset[current + horizon].edge_label_index
    return edge_label, edge_label_index


def update_batch(batch, batch_new, mode):
    if mode == 'replace':
        for i in range(len(batch_new.node_states)):
            batch_new.node_states[i] = batch.node_states[i].detach().cpu()
        batch_new.node_degree_existing = batch.node_degree_existing.detach(

        ).cpu()
        return batch_new
    elif mode == 'concat':
        keys = ['edge_feature', 'edge_index', 'edge_time']
        for key in keys:
            dim = 1 if 'index' in key else 0
            batch[key] = torch.cat([batch[key], batch_new[key]], dim=dim)
        return batch


def retrieve_idle_nodes(graph_snapshots, k: int) -> np.ndarray:
    """Get a list of nodes who remains inactive after snapshot k."""
    active_after_k = []
    for graph in graph_snapshots[k:]:
        src = np.unique(graph.edge_index[0].cpu().numpy())
        dst = np.unique(graph.edge_index[1].cpu().numpy())
        active_after_k.extend([src, dst])
    active_after_k = np.unique(np.concatenate(active_after_k))
    num_nodes = graph_snapshots[0].node_feature.shape[0]
    idle = np.setdiff1d(np.arange(num_nodes), active_after_k)
    return idle


@torch.no_grad()
def inference_epoch(model, dataset):
    target_nodes = retrieve_idle_nodes(dataset, int(0.1 * len(dataset)))
    writer = SummaryWriter()
    model.eval()
    rng = range(len(dataset) - cfg.transaction.horizon)
    for i in tqdm(rng, desc='Snapshot'):
        torch.cuda.empty_cache()
        if i == 0:
            batch = dataset[i].clone()
        else:
            # update raw features
            batch = update_batch(batch, dataset[i].clone(), mode='replace')
        # print(batch.node_degree_existing)
        batch.node_degree_new = node_degree(
            batch.edge_index,
            n=batch.node_degree_existing.shape[0])
        # batch.node_degree_new.to(cfg.device)

        batch.keep_ratio = train_utils.get_keep_ratio(
            existing=batch.node_degree_existing,
            new=batch.node_degree_new,
            mode=cfg.transaction.keep_ratio)
        batch.keep_ratio = batch.keep_ratio.unsqueeze(-1)
        batch.node_degree_existing += batch.node_degree_new

        # set edge labels
        edge_label, edge_label_index = get_edge_label(dataset, i,
                                                      cfg.transaction.horizon,
                                                      cfg.transaction.pred_mode)

        batch.edge_label = edge_label
        batch.edge_label_index = edge_label_index

        batch.to(torch.device(cfg.device))
        # move state to gpu
        for layer in range(len(batch.node_states)):
            if torch.is_tensor(batch.node_states[layer]):
                batch.node_states[layer] = batch.node_states[layer].to(
                    torch.device(cfg.device))
        pred, true = model(batch)

        target_node_states = batch.node_states[-1][target_nodes, :]
        l2 = torch.norm(target_node_states, p=2)
        writer.add_histogram('L2 norm of node embeddings', l2)
