"""
This script includes training/validating/testing procedures for the back
propagation through time (BPTT) training scheme.
"""
import logging
import time

import torch
from tqdm import tqdm

from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from graphgym.config import cfg
from graphgym.contrib.train import train_utils
from graphgym.loss import compute_loss
from graphgym.register import register_train
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.utils.stats import node_degree


def get_edge_label(dataset, current, horizon, mode):
    if mode == 'before':
        # target: {current+1, current+2, ..., current+horizon}
        edge_label = torch.cat([dataset[current + i].edge_label
                                for i in range(1, horizon + 1)], dim=0)
        edge_label_index = torch.cat([dataset[current + i].edge_label_index
                                      for i in range(1, horizon + 1)], dim=1)
    elif mode == 'at':
        # target: {current+horizon}
        edge_label = dataset[current + horizon].edge_label
        edge_label_index = dataset[current + horizon].edge_label_index
    else:
        raise ValueError
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
    with dummy_context_mgr() if train else torch.no_grad():
        if train:
            model.train()
        else:
            model.eval()
        time_start = time.time()

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # initialize loss tracker.
        accumulated_loss = torch.tensor(0.0).to(torch.device(cfg.device))
        rng = range(len(dataset) - cfg.transaction.horizon)
        for i in tqdm(rng, desc='Snapshot', leave=False):
            # using full history
            if cfg.transaction.history == 'full':
                if i == 0:
                    batch_history = dataset[i].clone()
                else:
                    batch_history = update_batch(batch_history,
                                                 dataset[i].clone(),
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

                batch.node_degree_new = node_degree(
                    batch.edge_index,
                    n=batch.node_degree_existing.shape[0])

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
            edge_label, edge_label_index = get_edge_label(
                dataset, i, cfg.transaction.horizon, cfg.transaction.pred_mode)

            batch.edge_label = edge_label
            batch.edge_label_index = edge_label_index

            if not train:
                # The batch for evaluation purpose only.
                eval_batch = batch.clone()

            if cfg.transaction.loss == 'meta':
                raise NotImplementedError
            elif cfg.transaction.loss == 'supervised':
                batch.to(torch.device(cfg.device))
                # move state to gpu
                for layer in range(len(batch.node_states)):
                    if torch.is_tensor(batch.node_states[layer]):
                        batch.node_states[layer] = batch.node_states[layer].to(
                            torch.device(cfg.device))
                pred, true = model(batch)
            else:
                raise ValueError(f'Invalid loss: {cfg.transaction.loss}')

            loss, pred_score = compute_loss(pred, true)
            accumulated_loss += loss

            if train and ((i + 1) % cfg.train.tbptt_freq == 0):
                # back prop on accumulated loss.
                # print(i, 'run bptt.')
                accumulated_loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                torch.cuda.empty_cache()
                # reset the loss tracker.
                accumulated_loss.detach()
                del accumulated_loss
                accumulated_loss = torch.tensor(0.0).to(
                    torch.device(cfg.device))

            if report_rank_based_metric:
                mrr, rck1, rck3, rck10 = train_utils.report_rank_based_eval(
                    eval_batch, model)

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
            if (i + 1) % cfg.train.tbptt_freq != 0:
                # use the remaining gradients, useful when num_snapshots does
                # not divide cfg.train.tbptt_freq.
                # print(i, 'run bptt (clean up).')
                accumulated_loss.backward()
                optimizer.step()
            scheduler.step()


def train_truncated_bptt(loggers, loaders, model, optimizer, scheduler,
                         datasets, **kwargs):
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

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


register_train('tbptt', train_truncated_bptt)
