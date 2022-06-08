"""
This file contains the pipeline for training the online / dynamic version of
transaction GNN.
"""
import torch
import time
import logging

import numpy as np

from deepsnap.batch import Batch

from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt

from graphgym.register import register_train


def make_incremental_graphs(G, trans_period_length: int):
    """Converts the entire graph / batch object into a sequence
        of incremental graphs.
    G_incr contains both features and targets required to train the model:
        (i) New transactions within trans_period (t - X, t) as features.
        (ii) Transactions in the fore_period (t, t + F),
            combined with negative samples, as targets.

    NOTE: we did not add E_prev into this module, E_prev should be incorporated
        in the training loop.
    """
    forecast_horizon = cfg.link_pred_spec.forecast_horizon
    data_start = torch.min(G.edge_time_raw)
    # Ignore the last few days due to lack of targets.
    data_end = torch.max(G.edge_time_raw) - torch.scalar_tensor(
        86400 * forecast_horizon, dtype=torch.int32)
    period_range = torch.arange(data_start, data_end,
                                step=trans_period_length * 86400).to(
        torch.device(cfg.device))

    for trans_start in period_range:
        # ==== construct the incremental graph ====
        trans_end = trans_start + torch.scalar_tensor(
            trans_period_length * 86400, dtype=torch.int32
            ).to(torch.device(cfg.device))
        # TODO: need to think about the range, open interval or closed interval.
        # Get edges in the transaction period.
        incr_edge_mask = torch.logical_and(
            (G.edge_time_raw >= trans_start).bool(),
            (G.edge_time_raw < trans_end).bool()
        ).bool().to(torch.device(cfg.device))

        # ==== construct what to be predicted ====
        fore_start = trans_end
        fore_end = fore_start + torch.scalar_tensor(
            forecast_horizon * 86400,
            dtype=torch.int32
        ).to(torch.device(cfg.device))
        forecast_mask = torch.logical_and(
            (G.edge_time_raw >= fore_start).bool(),
            (G.edge_time_raw < fore_end).bool()
        ).bool().to(torch.device(cfg.device))
        # Get positive edges.
        p_edge_index = G.edge_index[:, forecast_mask]
        num_pos = p_edge_index.shape[1]

        # Sample the same number of non-existent edges as negative samples.
        neg_idx = torch.randperm(torch.sum(G.edge_label == 0))[:num_pos]
        all_non_exist_edge_samples = G.edge_label_index[:, G.edge_label == 0]
        n_edge_index = all_non_exist_edge_samples[:, neg_idx]
        num_neg = n_edge_index.shape[1]

        # Combine positive and negative samples.
        edge_label_index = torch.cat(
            (p_edge_index, n_edge_index), axis=1
        ).to(torch.device(cfg.device))
        # Create labels.
        edge_label = torch.cat(
            (torch.ones(num_pos), torch.zeros(num_neg))
        ).long().to(torch.device(cfg.device))

        # ==== additional features to assist predicting edge_label_index ====
        # TODO: potentially use positional encoding for this time_delta.
        f_time_delta = fore_end - G.edge_time_raw[incr_edge_mask]

        # Put everything together into one single batch.
        G_incr = Batch(
            # Un-modified fields:
            batch=G.batch,
            directed=G.directed,
            node_feature=G.node_feature,
            node_label_index=G.node_label_index,
            task=G.task,
            # New transactions for message passing in (trans_start, trans_end)
            edge_feature=G.edge_feature[incr_edge_mask, :],
            edge_index=G.edge_index[:, incr_edge_mask],
            edge_time=G.edge_time[incr_edge_mask],
            edge_time_raw=G.edge_time_raw[incr_edge_mask],
            edge_time_delta=f_time_delta,
            # Target transactions to be predicted
            # in (trans_end=fore_start, fore_end)
            edge_label=edge_label,
            edge_label_index=edge_label_index
        )
        yield G_incr.to(torch.device(cfg.device))


def train_epoch(logger, loader, model, optimizer=None,
                scheduler=None, training=True):
    if training:
        model.train()
    else:
        model.eval()
    time_start = time.time()
    # TODO: retrieve these variables from cfg instead.
    emb_dim = 256
    incr_period = 7  # unit = day(s).
    for batch in loader:
        batch = batch.to(torch.device(cfg.device))

        true_list, pred_score_list, loss_list = [], [], []
        # Set initial embedding to zeros.
        emb_prev = torch.zeros((batch.num_nodes, emb_dim)).float().to(
            torch.device(cfg.device))
        for G_incr in make_incremental_graphs(batch, incr_period):
            if training:
                optimizer.zero_grad()
            # Append previous node embeddings to node features.
            # G_incr.node_feature = torch.cat(
            #     (G_incr.node_feature, emb_prev), dim=-1
            # ).to(torch.device(cfg.device))
            G_incr.node_feature = emb_prev
            pred, true = model(G_incr)
            loss, pred_score = compute_loss(pred, true)
            if training:
                loss.backward()
                optimizer.step()
            # Add records for later summary.
            true_list.append(true.detach().cpu())
            pred_score_list.append(pred_score.detach().cpu())
            loss_list.append(loss.item())
        logger.update_stats(true=torch.cat(true_list).reshape(-1, ),
                            pred=torch.cat(pred_score_list).reshape(-1, ),
                            loss=np.mean(loss_list),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


# def eval_epoch(logger, loader, model):
#     raise NotImplementedError
#     model.eval()
#     time_start = time.time()
#     for batch in loader:
#         batch = batch.to(torch.device(cfg.device))
#
#         forecast_horizon = cfg.link_pred_spec.forecast_horizon
#         offset = torch.scalar_tensor(86400 * forecast_horizon, dtype=torch.int32)
#         today_start = torch.min(batch.edge_time_raw) + offset
#         today_end = torch.max(batch.edge_time_raw) - offset
#         # The range of today, on which predictions are made.
#         today_range = torch.arange(
#             today_start, today_end, step=cfg.link_pred_spec.forecast_frequency * 86400
#         ).to(torch.device(cfg.device))
#
#         true_list, pred_score_list, loss_list = [], [], []
#         for today in today_range:
#             sample_batch = get_sample(batch, today, forecast_horizon).to(torch.device(cfg.device))
#             pred, true = model(sample_batch)
#             loss, pred_score = compute_loss(pred, true)
#             # Add records for later summary.
#             true_list.append(true.detach().cpu())
#             pred_score_list.append(pred_score.detach().cpu())
#             loss_list.append(loss.item())
#
#         # print(true.sum())
#         # print((pred>0.5).long().sum())
#         logger.update_stats(true=torch.cat(true_list).reshape(-1,),
#                             pred=torch.cat(pred_score_list).reshape(-1,),
#                             loss=np.mean(loss_list),
#                             lr=0,
#                             time_used=time.time() - time_start,
#                             params=cfg.params)
#         time_start = time.time()


def train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model,
                    optimizer, scheduler, training=True)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            # Report both validation and testing set performance.
            for i in range(1, num_splits):
                # eval_epoch(loggers[i], loaders[i], model)
                train_epoch(loggers[i], loaders[i], model, training=False)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


register_train('roland_bsi_online', train)
