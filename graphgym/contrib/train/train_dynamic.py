import torch
import time
import logging

from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from graphgym.utils.stats import node_degree

from graphgym.register import register_train

import pdb


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


def train_epoch(logger, model, optimizer, scheduler, dataset, train=True):
    model.train()
    time_start = time.time()
    for i in range(len(dataset) - cfg.transaction.horizon):
        optimizer.zero_grad()
        batch = dataset[i].clone()
        pdb.set_trace()
        # split into support edges
        # batch.edge_index =

        batch.node_degree_new = node_degree(batch.edge_index,
                                        n=batch.node_degree_existing.shape[0])

        edge_label, edge_label_index = get_edge_label(dataset, i,
                                                      cfg.transaction.horizon,
                                                      cfg.transaction.pred_mode)
        batch.edge_label = edge_label
        batch.edge_label_index = edge_label_index
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        if train:
            loss.backward()
            optimizer.step()
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
                    datasets[0], train=True)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                train_epoch(loggers[i], model, optimizer, scheduler,
                            datasets[i], train=False)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


register_train('dynamic', train_example)
