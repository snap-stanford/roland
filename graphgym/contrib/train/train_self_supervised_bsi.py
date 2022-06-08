"""
This file contains the training pipeline for self-supervised link prediction task.
"""
import torch
import time
import logging
from typing import Generator

import numpy as np

from torch_geometric.utils import negative_sampling

import deepsnap
from deepsnap.graph import Graph
from deepsnap.batch import Batch

from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt

from graphgym.register import register_train


def generate_self_supervised_samples(
    graph: "deepshap.graph.Graph",
    forecast_horizon: int
) -> Generator:
    """
    Extracts self-supervised learning signals for link prediction tasks from a graph.

    Functionality:
        ```
        for today in SCOPE:
            yield
                1. prior_mask: [data.num_edges, ] boolean, data[prior_mask] gives transaction graph <= today.
                2. sample_edge_index:
                    { (payer, payee) : exists transaction from payer to payee in (today, today + forecast_range] }
                    U { (payer, payee) : transaction from payer to payee in (today, today + forecast_range] }
                3. label: 1, 1, 1, ...., 0, 0, 0.
                4. today (for debugging purpose).
        ```
        Usage:
        1. Compute embedding using data[prior_mask].
        2. P = Similarity( embedding[sample_edge_index[0, :]], embedding[sample_edge_index[1, :]] )
        3. L = CrossEntropy(P, label)
    Args:
        graph: a deepsnap graph object of transaction graphs.
        forecast_horizon: an integer indicating the forecasting horizon in terms of DAYS.
            prediction task would be a binary classification: whether there will be transactions between
            company X and company X in next {forecast_horizon} days, i.e., (today, today + forecast_horizon].
            A greater value of forecast horizon leads to more positive samples.
    Returns:
        A generator of (prior_mask, sample_edge_index, labels, today)
    """
    # Number of days at the beginning and end of the dataset to be dropped.
    # Drop beginning because there are not enough features to construct embeddings.
    # Drop end because there are not enough labels.
    # By default, drop `forecast_horizon` days at each end.
    offset = torch.scalar_tensor(86400 * forecast_horizon, dtype=torch.int32)
    today_start = torch.min(graph.edge_time_raw) + offset
    today_end = torch.max(graph.edge_time_raw) - offset

    # Range of forecasting, convert into seconds as in timestamp.
    f = torch.scalar_tensor(forecast_horizon * 86400, dtype=torch.int32).to(torch.device(cfg.device))

    # The range of today, on which predictions are made.
    today_range = torch.arange(today_start, today_end, step=86400).to(torch.device(cfg.device))

    # Generate batches based on `today`.
    for today in today_range:
        # Only keep transaction (edge) prior to (inclusive) today.
        # This prior mask indicates chronological property of all edges in the original graph,
        # including edges for all of training, validation and testing purpose.
        prior_mask = (graph.edge_time_raw <= today).bool()

        # # Construct positive samples within (today, today + forecast] range.
        # forecast_mask = torch.logical_and(
        #     (graph.edge_time_raw > today).bool(),
        #     (graph.edge_time_raw <= today + f).bool()
        # )
        # positive_edge_index = graph.edge_index[:, forecast_mask]
        # num_positive = positive_edge_index.shape[1]
        #
        # # Construct negative samples.
        # negative_edge_index = negative_sampling(
        #     positive_edge_index,
        #     num_nodes=graph.num_nodes,
        #     num_neg_samples=num_positive
        # )
        #
        # num_negative = negative_edge_index.shape[1]

        # Construct positive samples within (today, today + forecast] range.
        # Retrieve edge time only for edges of {train, val, test} set.
        # c_* stands for current_*
        c_edge_index = graph.edge_label_index  # which edges are in the current {train, val, test} batch.
        # graph.edge_split_index contains the natural number indices of edges in the current batch.
        c_edge_time_raw = graph.edge_time_raw[graph.edge_split_index]
        forecast_mask = torch.logical_and(
            (c_edge_time_raw > today).bool(),
            (c_edge_time_raw <= today + f).bool()
        )
        positive_edge_index = c_edge_index[:, forecast_mask]
        num_positive = positive_edge_index.shape[1]

        # Construct negative samples.
        negative_edge_index = negative_sampling(
            positive_edge_index,
            num_nodes=graph.num_nodes,
            num_neg_samples=num_positive
        )

        num_negative = negative_edge_index.shape[1]

        # Concatenate positive and negative samples.
        sample_edge_index = torch.cat((positive_edge_index, negative_edge_index), axis=1).to(torch.device(cfg.device))
        # Construct labels, positive first and then negative.
        labels = torch.zeros(num_positive + num_negative).float()
        labels[:num_positive] = 1.
        labels = labels.to(torch.device(cfg.device))

        # Use a generator might use memory more efficiently.
        # returns today as well for debugging purpose.
        yield prior_mask, sample_edge_index, labels, today


def get_prior_graph(G, edge_time_mask):
    # The original edge_time_mask is boolean for all edges in the original graph.
    # Convert the mask to be applied on the current {train, val, test} batch.
    edge_mask_current_batch = edge_time_mask[G.edge_split_index]
    # Retrieve the graph prior to today, based on edge_time_mask.
    # Only return necessary attributes.
    return Batch(
        directed=G.directed,
        edge_feature=G.edge_feature[edge_time_mask, :],
        edge_index=G.edge_index[:, edge_time_mask],
        node_feature=G.node_feature
    )


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    # Need other ways of splitting, use full batch instead.
    for batch in loader:  # Graph level mini-batch.
        batch = batch.to(torch.device(cfg.device))
        # One batch corresponds to one graph.
        # In this case, there is only one batch in the loader.
        true_list, pred_score_list, loss_list = [], [], []
        for edge_time_mask, sample_edges, true, today \
                in generate_self_supervised_samples(batch, forecast_horizon=7):
            # edge_time_mask: boolean on all edges of the original graph, indicating whether a particular edge
            # occurred before or on today.
            prior_graph = get_prior_graph(batch, edge_time_mask)
            prior_graph.to(torch.device(cfg.device))
            # TODO: Optional, add edge level mini-batch here.
            optimizer.zero_grad()
            # Add positive and negative samples to the prior graph.
            prior_graph.edge_label_index = sample_edges
            prior_graph.edge_label = true
            pred, true = model(prior_graph)
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
        true_list, pred_score_list, loss_list = [], [], []
        for edge_time_mask, sample_edges, true, today \
                in generate_self_supervised_samples(batch, forecast_horizon=7):
            # Retrieve transactions prior to today.
            prior_graph = get_prior_graph(batch, edge_time_mask)
            prior_graph.to(torch.device(cfg.device))
            # Add positive edges in (today, today + forecast range] altogether with sampled negative edges.
            prior_graph.edge_label_index = sample_edges
            prior_graph.edge_label = true
            pred, true = model(prior_graph)
            loss, pred_score = compute_loss(pred, true)
            # Add records for later summary.
            true_list.append(true.detach().cpu())
            pred_score_list.append(pred_score.detach().cpu())
            loss_list.append(loss.item())

        # print(true.sum())
        # print((pred>0.5).long().sum())
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


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


# register_train('bsi_self_supervised', train)
