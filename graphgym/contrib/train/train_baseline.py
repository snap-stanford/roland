"""
A baseline training scheme similar to the training scheme in evolveGCN paper.
"""
import datetime
import logging
import time

import torch
from graphgym.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from graphgym.config import cfg
from graphgym.contrib.train import train_utils
from graphgym.loss import compute_loss
from graphgym.register import register_train
from graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch
from graphgym.utils.stats import node_degree
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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


def train_epoch(logger, model, optimizer, scheduler, dataset, writer, train=True):
    with dummy_context_mgr() if train else torch.no_grad():
        if train:
            model.train()
        else:
            model.eval()
        time_start = time.time()

        mrr_lst, rck1_lst, rck3_lst, rck10_lst = [], [], [], []

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        # track cumulative loss over all snapshots.
        accumulated_loss = torch.tensor(0.0).to(torch.device(cfg.device))

        rng = range(len(dataset) - cfg.transaction.horizon)
        for i in tqdm(rng, desc='Snapshot'):
            optimizer.zero_grad()
            torch.cuda.empty_cache()

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

            # set edge labels
            edge_label, edge_label_index = get_edge_label(dataset, i,
                                                          cfg.transaction.horizon,
                                                          cfg.transaction.pred_mode)

            batch.edge_label = edge_label
            batch.edge_label_index = edge_label_index

            eval_batch = batch.clone()

            batch = train_utils.move_batch_to_device(batch, cfg.device)
            pred, true = model(batch)

            loss, pred_score = compute_loss(pred, true)
            accumulated_loss += loss

            if train and ((i + 1) % cfg.train.tbptt_freq == 0):
                accumulated_loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                torch.cuda.empty_cache()
                # reset the loss tracker.
                accumulated_loss.detach()
                del accumulated_loss
                accumulated_loss = torch.tensor(0.0).to(
                    torch.device(cfg.device))

            # Compute rank based metrics for the current snapshot.
            mrr, rck1, rck3, rck10 = train_utils.report_rank_based_eval(
                eval_batch, model,
                num_neg_per_node=cfg.experimental.rank_eval_multiplier)
            mrr_lst.append(mrr)
            rck1_lst.append(rck1)
            rck3_lst.append(rck3)
            rck10_lst.append(rck10)

            writer.add_scalar('loss', loss, i)
            writer.add_scalar('mrr', mrr, i)
            writer.add_scalar('rck1', rck1, i)
            writer.add_scalar('rck3', rck3, i)
            writer.add_scalar('rck10', rck10, i)

            logger.update_stats(true=true.detach().cpu(),
                                pred=pred_score.detach().cpu(),
                                loss=loss.item(),
                                lr=scheduler.get_last_lr()[0],
                                time_used=time.time() - time_start,
                                params=cfg.params,
                                mrr=mrr, rck1=rck1, rck3=rck3, rck10=rck10)

            time_start = time.time()

        if train and cfg.train.tbptt_freq > 1:
            if (i + 1) % cfg.train.tbptt_freq != 0:
                # use the remaining gradients, useful when num_snapshots does
                # not divide cfg.train.tbptt_freq.
                accumulated_loss.backward()
                optimizer.step()

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

    t = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    # directory to store tensorboard files of this run.
    out_dir = t + '_' + cfg.out_dir.replace('/', '\\')
    # dir to store all run outputs for the entire batch.
    run_dir = 'runs_' + cfg.remark

    print(f'Tensorboard directory: {out_dir}')

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_writer = SummaryWriter(f'./{run_dir}/{out_dir}/train_{cur_epoch}')
        val_writer = SummaryWriter(f'./{run_dir}/{out_dir}/val_{cur_epoch}')
        test_writer = SummaryWriter(f'./{run_dir}/{out_dir}/test_{cur_epoch}')

        train_epoch(loggers[0], model, optimizer, scheduler,
                    datasets[0], train_writer, train=True)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i, writer in zip(range(1, num_splits), [val_writer, test_writer]):
                train_epoch(loggers[i], model, optimizer, scheduler,
                            datasets[i], writer, train=False)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))


register_train('baseline', train_example)
