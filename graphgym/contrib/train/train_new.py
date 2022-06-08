"""
This script includes training/validating/testing procedures for rolling scheme.
"""
import logging
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from graphgym.config import cfg
from graphgym.contrib.train import plot_utils
from graphgym.contrib.train import train_utils
from graphgym.loss import compute_loss
from graphgym.register import register_train
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.utils.stats import node_degree


def create_eval_label(eval_batch):
    # Get positive edge indices.
    edge_index = eval_batch.edge_label_index[:, eval_batch.edge_label == 1].to(
        "cpu")
    # idx = N * i + j
    idx = (edge_index[0] * eval_batch.num_nodes + edge_index[1]).to("cpu")

    # Generate negative edges, get senders of positive edges.
    senders = torch.unique(edge_index[0]).detach().cpu()
    # Consider these senders as users, sample negative edges for each user.
    multiplier = 10  # (approximately) how many negative edges for each sender.
    senders = senders.repeat_interleave(multiplier)
    random_receivers = torch.tensor(np.random.choice(
        eval_batch.num_nodes, len(senders), replace=True))

    perm = (senders * eval_batch.num_nodes + random_receivers)
    mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
    perm = perm[~mask]  # Filter out false negative edges.
    row = perm // eval_batch.num_nodes
    col = perm % eval_batch.num_nodes
    neg_edge_index = torch.stack([row, col], dim=0).long()

    new_edge_label_index = torch.cat((edge_index, neg_edge_index),
                                     dim=1).long()
    new_edge_label = torch.cat((torch.ones(edge_index.shape[1]),
                                torch.zeros(neg_edge_index.shape[1])
                                ), dim=0).long()

    eval_batch.edge_label_index = new_edge_label_index.to(
        torch.device(cfg.device))
    eval_batch.edge_label = new_edge_label.to(torch.device(cfg.device))

    eval_batch.senders = senders
    return eval_batch


# --------------------------------------------------------------------------- #
# Training Modules.
# --------------------------------------------------------------------------- #


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


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def train_epoch(logger, model, optimizer, scheduler, dataset, train=True,
                report_rank_based_metric=False):
    """A single epoch of training, validating or testing.
    """
    # with dummy_context_mgr() if train else torch.no_grad():
    with dummy_context_mgr() if train else torch.no_grad():
        if train:
            model.train()
        else:
            model.eval()
        time_start = time.time()

        mrr_lst, rck1_lst, rck3_lst, rck10_lst = [], [], [], []

        rng = range(len(dataset) - cfg.transaction.horizon)
        if train:
            # Only train on a smaller subset of periods.
            if cfg.experimental.restrict_training_set != -1:
                assert cfg.experimental.restrict_training_set > 1
                num_periods = int(cfg.experimental.restrict_training_set)
                # Option 1: take the first k.
                rng = list(range(num_periods))
                # Option 2: randomly pick k in chronological order.
                # rng = [0] + random.sample(rng, k=num_periods-1)
                # rng = sorted(list(set(rng)))

        # for i in range(len(dataset) - cfg.transaction.horizon):
        for i in tqdm(rng, desc='Snapshot'):
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            # using full history
            if cfg.transaction.history == 'full':
                if i == 0:
                    batch_history = dataset[i]
                else:
                    batch_history = update_batch(batch_history, dataset[i],
                                                 mode='concat')
                batch = batch_history.clone()
            # using rolling history
            elif cfg.transaction.history == 'rolling':
                if i == 0:
                    batch = dataset[i].clone()
                else:
                    # update raw features
                    batch = update_batch(batch, dataset[i].clone(),
                                         mode='replace')
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

            else:
                raise ValueError(
                    f'Unsupported training mode: {cfg.transaction.history}')

            # set edge labels
            edge_label, edge_label_index = get_edge_label(dataset, i,
                                                          cfg.transaction.horizon,
                                                          cfg.transaction.pred_mode)

            batch.edge_label = edge_label
            batch.edge_label_index = edge_label_index

            # Uncomment to use time encoding for time positional encoding.
            # pred_time = min(dataset[i+1].edge_time)
            # batch.edge_time_delta = pred_time - batch.edge_time

            # copy a batch for evaluation purpose, since model.forward will
            # modify attributes in batch.
            eval_batch = batch.clone()

            if cfg.transaction.loss == 'meta':
                raise NotImplementedError
            elif cfg.transaction.loss == 'supervised':
                batch = train_utils.move_batch_to_device(batch, cfg.device)
                pred, true = model(batch)
            else:
                raise ValueError(f'Invalid loss: {cfg.transaction.loss}')

            loss, pred_score = compute_loss(pred, true)
            if train:
                loss.backward()
                optimizer.step()

            if report_rank_based_metric:
                # Compute rank based metrics for the current snapshot.
                mrr, rck1, rck3, rck10 = train_utils.report_rank_based_eval(
                    eval_batch, model,
                    num_neg_per_node=cfg.experimental.rank_eval_multiplier)
                mrr_lst.append(mrr)
                rck1_lst.append(rck1)
                rck3_lst.append(rck3)
                rck10_lst.append(rck10)

                logger.update_stats(true=true.detach().cpu(),
                                    pred=pred_score.detach().cpu(),
                                    loss=loss.item(),
                                    lr=scheduler.get_last_lr()[0],
                                    time_used=time.time() - time_start,
                                    params=cfg.params,
                                    mrr=mrr, rck1=rck1, rck3=rck3, rck10=rck10)
            else:
                logger.update_stats(true=true.detach().cpu(),
                                    pred=pred_score.detach().cpu(),
                                    loss=loss.item(),
                                    lr=scheduler.get_last_lr()[0],
                                    time_used=time.time() - time_start,
                                    params=cfg.params)

            time_start = time.time()
        if train:
            scheduler.step()


def train_example(loggers, loaders, model, optimizer, scheduler, datasets,
                  **kwargs):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], model, optimizer, scheduler,
                    datasets[0], train=True,
                    report_rank_based_metric=False)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                train_epoch(loggers[i], model, optimizer, scheduler,
                            datasets[i], train=False,
                            report_rank_based_metric=True)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    if cfg.experimental.visualize_gnn_layer:
        # Analysis the attention on the validation set.
        # Save attention weights to disk for later visualization.
        cur_time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        for i in range(cfg.gnn.layers_mp):
            getattr(model.mp, f'layer{i}').layer.model.save_att_weights(
                f'./{cur_time}_att_weights_layer{i}')

        # Run an extra epoch to get attention.
        # TODO: choose the dataset you want!
        train_epoch(loggers[1], model, optimizer, scheduler,
                    datasets[1], train=False,
                    report_rank_based_metric=False)

        # Visualize attention weights.
        for i in range(cfg.gnn.layers_mp):
            plot_utils.visualize_attention(datasets[1],
                                           att_path=f'./{cur_time}_att_weights_layer{i}',
                                           fig_path=f'./{cur_time}_plots_layer{i}')

        # Lastly, plot something about the dataset.
        plot_utils.visualize_dataset(datasets,
                                     fig_path=f'./{cur_time}_plots_datasets')

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


register_train('new', train_example)
