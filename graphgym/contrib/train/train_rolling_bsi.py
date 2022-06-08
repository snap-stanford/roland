"""
This file contains the training pipeline for self-supervised link prediction task using BSI dataset.
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


def get_sample(G, today: int, forecast_horizon: int):
    """Constructs the training batch from original graph (batch) G.

    Args:
        G: the entire dataset, either a Graph object or a Batch object.
        today: an integer of timestamp indicating today.
        forecast_horizon: how many days into the future to be predicted.

    The batch returned by this method contains:
        (i) The graph restricted to time period (-infty, today] for message passing.
        (ii) Positive edges in (today, today + forecast horizon] and the same amount of negative edges.

    See comments below for detailed explanations.
    """
    device = torch.device(cfg.device)
    # ==== construct edges for message passing (MP) ====
    # Use all positive edges in (-infty, today] for message passing and embedding construction.
    prior_edge_mask = (G.edge_time_raw <= today).bool().to(device)

    # ==== construct edges to be predicted ====
    # Get edges in (today, today + forecast horizon] as positive samples.
    f = torch.scalar_tensor(forecast_horizon * 86400, dtype=torch.int32).to(device)
    forecast_mask = torch.logical_and(
        (G.edge_time_raw > today).bool(),
        (G.edge_time_raw <= today + f).bool()
    )
    p_edge_index = G.edge_index[:, forecast_mask]
    num_pos = p_edge_index.shape[1]

    # Randomly sample the same number of non-existent edges as negative samples.
    neg_idx = torch.randperm(torch.sum(G.edge_label == 0))[:num_pos]
    all_non_exist_edge_samples = G.edge_label_index[:, G.edge_label == 0]
    n_edge_index = all_non_exist_edge_samples[:, neg_idx]
    num_neg = n_edge_index.shape[1]

    # Combine positive and negative samples.
    edge_label_index = torch.cat((p_edge_index, n_edge_index), axis=1).to(device)
    # Create labels.
    edge_label = torch.cat(
        (torch.ones(num_pos), torch.zeros(num_neg))
    ).long().to(device)

    f_time_delta = today + f - G.edge_time_raw[prior_edge_mask]

    return Batch(
        batch=G.batch,
        directed=G.directed,
        edge_feature=G.edge_feature[prior_edge_mask, :],  # for all edges in (-infty, today] for MP.
        edge_index=G.edge_index[:, prior_edge_mask],  # for all edges in (-infty, today] for MP.
        edge_label=edge_label,  # labels of edges in (today, today + F] union non-existent edges for prediction.
        edge_label_index=edge_label_index,  # edges in (today, today + F] union non-existent edges for prediction.
        edge_time=G.edge_time[prior_edge_mask],  # time of edges in (-infty, today], will not be used.
        edge_time_raw=G.edge_time_raw[prior_edge_mask],  # time of edges in (-infty, today], will not be used.
        edge_time_delta=f_time_delta,
        node_feature=G.node_feature,
        node_label_index=G.node_label_index,
        task=G.task
    )


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        batch = batch.to(torch.device(cfg.device))

        forecast_horizon = cfg.link_pred_spec.forecast_horizon
        offset = torch.scalar_tensor(86400 * forecast_horizon, dtype=torch.int32)
        today_start = torch.min(batch.edge_time_raw) + offset
        today_end = torch.max(batch.edge_time_raw) - offset
        # The range of today, on which predictions are made.
        today_range = torch.arange(
            today_start, today_end, step=cfg.link_pred_spec.forecast_frequency * 86400
        ).to(torch.device(cfg.device))

        true_list, pred_score_list, loss_list = [], [], []
        for today in today_range:
            sample_batch = get_sample(batch, today, forecast_horizon).to(torch.device(cfg.device))
            optimizer.zero_grad()
            pred, true = model(sample_batch)
            loss, pred_score = compute_loss(pred, true)
            loss.backward()
            optimizer.step()
            # Add records for later summary.
            true_list.append(true.detach().cpu())
            pred_score_list.append(pred_score.detach().cpu())
            loss_list.append(loss.item())
        logger.update_stats(true=torch.cat(true_list).reshape(-1,),
                            pred=torch.cat(pred_score_list).reshape(-1,),
                            loss=np.mean(loss_list),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch = batch.to(torch.device(cfg.device))

        forecast_horizon = cfg.link_pred_spec.forecast_horizon
        offset = torch.scalar_tensor(86400 * forecast_horizon, dtype=torch.int32)
        today_start = torch.min(batch.edge_time_raw) + offset
        today_end = torch.max(batch.edge_time_raw) - offset
        # The range of today, on which predictions are made.
        today_range = torch.arange(
            today_start, today_end, step=cfg.link_pred_spec.forecast_frequency * 86400
        ).to(torch.device(cfg.device))

        true_list, pred_score_list, loss_list = [], [], []
        for today in today_range:
            sample_batch = get_sample(batch, today, forecast_horizon).to(torch.device(cfg.device))
            pred, true = model(sample_batch)
            loss, pred_score = compute_loss(pred, true)
            # Add records for later summary.
            true_list.append(true.detach().cpu())
            pred_score_list.append(pred_score.detach().cpu())
            loss_list.append(loss.item())

        # print(true.sum())
        # print((pred>0.5).long().sum())
        logger.update_stats(true=torch.cat(true_list).reshape(-1,),
                            pred=torch.cat(pred_score_list).reshape(-1,),
                            loss=np.mean(loss_list),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def train(loggers, loaders, model, optimizer, scheduler, datasets=None):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            # Report both validation and testing set performance.
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


register_train('roland_bsi', train)
