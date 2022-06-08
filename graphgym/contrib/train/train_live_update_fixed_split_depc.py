# """
# The more realistic training pipeline.
# """
# import copy
# import datetime
# import logging
# import os
# from typing import Dict, List, Optional, Tuple

# import deepsnap
# import numpy as np
# import torch
# from graphgym.checkpoint import clean_ckpt
# from graphgym.config import cfg
# from graphgym.contrib.train import train_utils
# from graphgym.loss import compute_loss
# from graphgym.optimizer import create_optimizer, create_scheduler
# from graphgym.register import register_train
# from graphgym.utils.io import makedirs_rm_exist
# from graphgym.utils.stats import node_degree
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm


# def get_edge_label(dataset, current, horizon, mode):
#     if mode == 'before':
#         # target: {current+1, current+2, ..., current+horizon}
#         edge_label = torch.cat([dataset[current + i].edge_label
#                                 for i in range(1, horizon + 1)], dim=0)
#         edge_label_index = torch.cat([dataset[current + i].edge_label_index
#                                       for i in range(1, horizon + 1)], dim=1)
#     elif mode == 'at':
#         # target: {current+horizon}
#         edge_label = dataset[current + horizon].edge_label
#         edge_label_index = dataset[current + horizon].edge_label_index
#     else:
#         raise ValueError
#     return edge_label, edge_label_index


# def update_batch(batch, batch_new, mode):
#     if mode == 'replace':
#         for i in range(len(batch_new.node_states)):
#             batch_new.node_states[i] = batch.node_states[i].detach().cpu()
#         batch_new.node_degree_existing = batch.node_degree_existing.detach(

#         ).cpu()
#         return batch_new
#     elif mode == 'concat':
#         keys = ['edge_feature', 'edge_index', 'edge_time']
#         for key in keys:
#             dim = 1 if 'index' in key else 0
#             batch[key] = torch.cat([batch[key], batch_new[key]], dim=dim)
#         return batch


# @torch.no_grad()
# def average_state_dict(dict1: dict, dict2: dict, weight: float) -> dict:
#     # Average two model.state_dict() objects.
#     # out = (1-w)*dict1 + w*dict2
#     assert 0 <= weight <= 1
#     d1 = copy.deepcopy(dict1)
#     d2 = copy.deepcopy(dict2)
#     out = dict()
#     for key in d1.keys():
#         assert isinstance(d1[key], torch.Tensor)
#         param1 = d1[key].detach().clone()
#         assert isinstance(d2[key], torch.Tensor)
#         param2 = d2[key].detach().clone()
#         out[key] = (1 - weight) * param1 + weight * param2
#     return out


# def precompute_edge_degree_info(dataset):
#     """Pre-computes edge_degree_existing, edge_degree_new and keep ratio
#     at each snapshot. Inplace modifications.
#     """
#     num_nodes = dataset[0].node_feature.shape[0]
#     for t in tqdm(range(len(dataset)), desc='precompute edge deg info'):
#         if t == 0:
#             dataset[t].node_degree_existing = torch.zeros(num_nodes)
#         else:
#             dataset[t].node_degree_existing \
#                 = dataset[t - 1].node_degree_existing \
#                   + dataset[t - 1].node_degree_new

#         dataset[t].node_degree_new = node_degree(dataset[t].edge_index,
#                                                  n=num_nodes)

#         dataset[t].keep_ratio = train_utils.get_keep_ratio(
#             existing=dataset[t].node_degree_existing,
#             new=dataset[t].node_degree_new,
#             mode=cfg.transaction.keep_ratio)
#         dataset[t].keep_ratio = dataset[t].keep_ratio.unsqueeze(-1)


# @torch.no_grad()
# def get_task_batch(dataset: deepsnap.dataset.GraphDataset,
#                    today: int, tomorrow: int,
#                    prev_node_states: Optional[Dict[str, List[torch.Tensor]]]
#                    ) -> deepsnap.graph.Graph:
#     """
#     Construct batch required for the task (today, tomorrow). As defined in
#     batch's get_item method (used to get edge_label and get_label_index),
#     edge_label and edge_label_index returned would be different everytime
#     get_task_batch() is called.

#     Moreover, copy node-memories (node_states and node_cells) to the batch.
#     """
#     assert today < tomorrow < len(dataset)
#     # Get edges for message passing and prediction task.
#     batch = dataset[today].clone()
#     batch.edge_label = dataset[tomorrow].edge_label.clone()
#     batch.edge_label_index = dataset[tomorrow].edge_label_index.clone()

#     # Copy previous memory to the batch.
#     if prev_node_states is not None:
#         for key, val in prev_node_states.items():
#             copied = [x.detach().clone() for x in val]
#             setattr(batch, key, copied)

#     batch = train_utils.move_batch_to_device(batch, cfg.device)
#     return batch


# @torch.no_grad()
# def update_node_states(model, dataset, task: Tuple[int, int],
#                        prev_node_states: Optional[
#                            Dict[str, List[torch.Tensor]]]
#                        ) -> Dict[str, List[torch.Tensor]]:
#     """Perform the provided task and keep track of the latest node_states.

#     Example: task = (t, t+1),
#         the prev_node_states contains node embeddings at time (t-1).
#         the model perform task (t, t+1):
#             Input: (node embedding at t - 1, edges at t).
#             Output: possible transactions at t+1.
#         the model also generates node embeddings at t.

#     after doing task (t, t+1), node_states contains information
#     from snapshot t.
#     """
#     today, tomorrow = task
#     batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()
#     # Let the model modify batch.node_states (and batch.node_cells).
#     _, _ = model(batch)
#     # Collect the updated node states.
#     out = dict()
#     out['node_states'] = [x.detach().clone() for x in batch.node_states]
#     if isinstance(batch.node_cells[0], torch.Tensor):
#         out['node_cells'] = [x.detach().clone() for x in batch.node_cells]

#     return out


# def train_step(model, optimizer, scheduler, dataset,
#                task: Tuple[int, int],
#                prev_node_states: Optional[Dict[str, torch.Tensor]]
#                ) -> dict:
#     """
#     After receiving ground truth from a particular task, update the model by
#     performing back-propagation.
#     For example, on day t, the ground truth of task (t-1, t) has been revealed,
#     train the model using G[t-1] for message passing and label[t] as target.
#     """
#     optimizer.zero_grad()
#     torch.cuda.empty_cache()

#     today, tomorrow = task
#     model.train()
#     batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()

#     pred, true = model(batch)
#     loss, pred_score = compute_loss(pred, true)
#     loss.backward()
#     optimizer.step()

#     scheduler.step()
#     return {'loss': loss}


# def train_epoch(model, optimizer, scheduler, dataset,
#                 train_on_last_snapshot=True):
#     model.train()

#     optimizer.zero_grad()
#     torch.cuda.empty_cache()

#     # initialize loss tracker.
#     accumulated_loss = torch.tensor(0.0).to(torch.device(cfg.device))
#     rng = range(len(dataset) - cfg.transaction.horizon)
#     # iterate through all training snapshots.
#     for i in tqdm(rng, desc='Snapshot', leave=False):
#         if i == 0:
#             batch = dataset[i].clone()
#         else:
#             # update raw features
#             batch = update_batch(batch, dataset[i].clone(),
#                                  mode='replace')

#         batch.node_degree_new = node_degree(
#             batch.edge_index,
#             n=batch.node_degree_existing.shape[0])

#         batch.keep_ratio = train_utils.get_keep_ratio(
#             existing=batch.node_degree_existing,
#             new=batch.node_degree_new,
#             mode=cfg.transaction.keep_ratio)
#         batch.keep_ratio = batch.keep_ratio.unsqueeze(-1)
#         batch.node_degree_existing += batch.node_degree_new

#         # set edge labels
#         edge_label, edge_label_index = get_edge_label(
#             dataset, i, cfg.transaction.horizon, cfg.transaction.pred_mode)

#         batch.edge_label = edge_label
#         batch.edge_label_index = edge_label_index

#         batch.to(torch.device(cfg.device))
#         # move state to gpu
#         for layer in range(len(batch.node_states)):
#             if torch.is_tensor(batch.node_states[layer]):
#                 batch.node_states[layer] = batch.node_states[layer].to(
#                     torch.device(cfg.device))
#         pred, true = model(batch)

#         loss, pred_score = compute_loss(pred, true)
        
#         if train_on_last_snapshot:
#             if i == len(dataset) - cfg.transaction.horizon - 1:
#                 # Only train on the last training snapshot.
#                 accumulated_loss += loss
#         else:
#             accumulated_loss += loss

#     accumulated_loss.backward()
#     print(f'loss = {accumulated_loss}')
#     optimizer.step()
#     scheduler.step()
    
#     last_node_states = [s.detach().clone() for s in batch.node_states]
#     return last_node_states


# @torch.no_grad()
# def val_epoch(model, dataset, prev_node_states):
#     model.eval()
#     rng = range(len(dataset) - cfg.transaction.horizon)
#     # iterate through all training snapshots.
#     for i in tqdm(rng, desc='Validation Snapshots', leave=False):
#         if i == 0:
#             batch = dataset[i].clone()
#             batch.node_states = prev_node_states
#         else:
#             # update raw features
#             batch = update_batch(batch, dataset[i].clone(),
#                                  mode='replace')

#         batch.node_degree_new = node_degree(
#             batch.edge_index,
#             n=batch.node_degree_existing.shape[0])

#         batch.keep_ratio = train_utils.get_keep_ratio(
#             existing=batch.node_degree_existing,
#             new=batch.node_degree_new,
#             mode=cfg.transaction.keep_ratio)
#         batch.keep_ratio = batch.keep_ratio.unsqueeze(-1)
#         batch.node_degree_existing += batch.node_degree_new

#         # set edge labels
#         edge_label, edge_label_index = get_edge_label(
#             dataset, i, cfg.transaction.horizon, cfg.transaction.pred_mode)

#         batch.edge_label = edge_label
#         batch.edge_label_index = edge_label_index

#         batch.to(torch.device(cfg.device))
#         # move state to gpu
#         for layer in range(len(batch.node_states)):
#             if torch.is_tensor(batch.node_states[layer]):
#                 batch.node_states[layer] = batch.node_states[layer].to(
#                     torch.device(cfg.device))
#         pred, true = model(batch)
#         loss, pred_score = compute_loss(pred, true)
    
#     last_node_states = [s.detach().clone() for s in batch.node_states]
#     return last_node_states


# @torch.no_grad()
# def evaluate_step(model, dataset, task: Tuple[int, int],
#                   prev_node_states: Optional[Dict[str, List[torch.Tensor]]],
#                   fast: bool = False) -> dict:
#     """
#     Evaluate model's performance on task = (today, tomorrow)
#         where today and tomorrow are integers indexing snapshots.
#     """
#     today, tomorrow = task
#     model.eval()
#     batch = get_task_batch(dataset, today, tomorrow, prev_node_states).clone()

#     pred, true = model(batch)
#     loss, pred_score = compute_loss(pred, true)

#     if fast:
#         # skip MRR calculation for internal validation.
#         return {'loss': loss.item()}

#     mrr_batch = get_task_batch(dataset, today, tomorrow,
#                                prev_node_states).clone()

#     mrr = train_utils.report_baseline_MRR(mrr_batch, model)
#     # print(mrr)
#     return {'loss': loss.item(), 'mrr': mrr}


# def train_live_update(loggers, loaders, model, optimizer, scheduler, datasets,
#                       **kwargs):

#     for dataset in datasets:
#         # Sometimes edge degree info is already included in dataset.
#         if not hasattr(dataset[0], 'keep_ratio'):
#             precompute_edge_degree_info(dataset)
#     t = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

#     # directory to store tensorboard files of this run.
#     out_dir = cfg.out_dir.replace('/', '\\')
#     # dir to store all run outputs for the entire batch.
#     run_dir = 'runs_' + cfg.remark

#     print(f'Tensorboard directory: {out_dir}')
#     # If tensorboard directory exists, this config is in the re-run phase
#     # of run_batch, replace logs of previous runs with the new one.
#     makedirs_rm_exist(f'./{run_dir}/{out_dir}')
#     writer = SummaryWriter(f'./{run_dir}/{out_dir}')

#     # save a copy of configuration for later identifications.
#     with open(f'./{run_dir}/{out_dir}/config.yaml', 'w') as f:
#         cfg.dump(stream=f)

#     prev_node_states = None  # no previous state on day 0.
#     # {'node_states': [Tensor, Tensor], 'node_cells: [Tensor, Tensor]}

#     model_init = None  # for meta-learning only, a model.state_dict() object.

#     del optimizer, scheduler  # use new optimizers for training phase.
#     optimizer = create_optimizer(model.parameters())
#     scheduler = create_scheduler(optimizer)
#     # 1. Training phase for the fixed split.
#     for _ in tqdm(range(cfg.optim.max_epoch), desc='Training', leave=True):
#         # Only aim to predict for the last snapshot.
#         prev_node_states = train_epoch(model, optimizer, scheduler, datasets[0],
#                                        train_on_last_snapshot=True)

#     # 2. Validation phase for the fixed split with live-update.
#     # Pass the model through validation sets to update node_states.
#     prev_node_states = val_epoch(model, datasets[1], prev_node_states)
#     prev_node_states = {'node_states': [s.detach().clone() for s in prev_node_states]}
#     # 3. Test phase for the fixed split with live-update.
#     task_range = range(len(datasets[2]) - cfg.transaction.horizon)

#     MRR_list = []
#     for t in tqdm(task_range, desc='Testing', leave=True):
#         # current task: t --> t+1.
#         # (1) Evaluate model's performance on this task, at this time, the
#         # model has seen no information on t+1, this evaluation is fair.
#         perf = evaluate_step(model, datasets[2], (t, t + 1), prev_node_states, fast=False)
#         print(f'Test = {perf}')
#         writer.add_scalars('test', perf, t)
#         MRR_list.append(perf['mrr'])

#         # (2) Reveal the ground truth of task (t, t+1) and update the model
#         # to prepare for the next task.
#         del optimizer, scheduler  # use new optimizers.
#         optimizer = create_optimizer(model.parameters())
#         scheduler = create_scheduler(optimizer)

#         # best model's validation loss, training epochs, and state_dict.
#         best_model = {'val_loss': np.inf, 'train_epoch': 0, 'state': None}
#         # keep track of how long we have NOT update the best model.
#         best_model_unchanged = 0
#         # after not updating the best model for `tol` epochs, stop.
#         tol = cfg.train.internal_validation_tolerance

#         # internal training loop (intra-snapshot cross-validation).
#         # choose the best model using current validation set, prepare for
#         # next task.

#         if cfg.meta.is_meta and (model_init is not None):
#             # For meta-learning, start fine-tuning from the pre-computed
#             # initialization weight.
#             model.load_state_dict(copy.deepcopy(model_init))

#         tol = 0
#         for i in tqdm(range(1 + 1), desc='live update',
#                       leave=True):
#             # Start with the un-trained model (i = 0), evaluate the model.
#             internal_val_perf = evaluate_step(model, datasets[2],
#                                               (t, t + 1),
#                                               prev_node_states, fast=True)
#             val_loss = internal_val_perf['loss']

#             if val_loss < best_model['val_loss']:
#                 # replace the best model with the current model.
#                 best_model = {'val_loss': val_loss, 'train_epoch': i,
#                               'state': copy.deepcopy(model.state_dict())}
#                 best_model_unchanged = 0
#             else:
#                 # the current best model has dominated for these epochs.
#                 best_model_unchanged += 1

#             # if (i >= 2 * tol) and (best_model_unchanged >= tol):
#             if best_model_unchanged >= tol:
#                 # If the best model has not been updated for a while, stop.
#                 break
#             else:
#                 # Otherwise, keep training.
#                 train_perf = train_step(model, optimizer, scheduler,
#                                         datasets[2], (t, t + 1),
#                                         prev_node_states)
#                 writer.add_scalars('train', train_perf, t)

#         writer.add_scalar('internal_best_val', best_model['val_loss'], t)
#         writer.add_scalar('best epoch', best_model['train_epoch'], t)

#         # (3) Actually perform the update on training set to get node_states
#         # contains information up to time t.
#         # Use the best model selected from intra-snapshot cross-validation.
#         model.load_state_dict(best_model['state'])

#         if cfg.meta.is_meta:  # update meta-learning's initialization weights.
#             if model_init is None:  # for the first task.
#                 model_init = copy.deepcopy(best_model['state'])
#             else:  # for subsequent task, update init.
#                 if cfg.meta.method == 'moving_average':
#                     new_weight = cfg.meta.alpha
#                 elif cfg.meta.method == 'online_mean':
#                     new_weight = 1 / (t + 1)  # for t=1, the second item, 1/2.
#                 else:
#                     raise ValueError(f'Invalid method: {cfg.meta.method}')

#                 # (1-new_weight)*model_init + new_weight*best_model.
#                 model_init = average_state_dict(model_init,
#                                                 best_model['state'],
#                                                 new_weight)

#         prev_node_states = update_node_states(model, datasets[2], (t, t + 1),
#                                               prev_node_states)
#     writer.close()
    
#     weights = [g.num_edges for g in datasets[2]]
#     avg_mrr = np.mean(MRR_list)
#     weighted_avg_mrr = np.sum([w * m for w, m in zip(weights, MRR_list)]) / np.sum(weights)
#     print(f'avg MRR = {avg_mrr}, weighted avg MRR = {weighted_avg_mrr}')
    
#     breakpoint()

#     if cfg.train.ckpt_clean:
#         clean_ckpt()

#     logging.info('Task done, results saved in {}'.format(cfg.out_dir))


# register_train('live_update_fixed_split', train_live_update)
