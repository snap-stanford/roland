"""
A simple utility that generates performance report for different model on
different datasets.

This script works for live-update scheme only, use graphgym's native analyze
tools for rolling/fixed-split scheme.
"""
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from tqdm import tqdm


def squeeze_dict(old_dict: dict) -> dict:
    """Squeezes nested dictionary keys.
        Example: old_dict['key1'] = {'key2': 'hello'}.
        will generate new_dict['key1.key2'] = 'hello'.
    """
    new_dict = dict()
    for k1 in old_dict.keys():
        if isinstance(old_dict[k1], dict):
            for k2 in old_dict[k1].keys():
                new_key = k1 + '.' + k2
                new_dict[new_key] = old_dict[k1][k2]
        else:
            new_dict[k1] = old_dict[k1]
    return new_dict


def tabulate_events(logdir: str, variables: List[str]) -> pd.DataFrame:
    """
    Generates a pandas dataframe which contains experiment (runs) as its rows,
    the returned dataframe contains columns:
        (1) File name/path of that run.
        (2) Fields required in `variables' from corresponding config.yaml.
        (3) Test and validation set performance (MRR and Recall at k).
    """
    all_runs = list()
    count = 0  # count number of experiment runs processed.

    for run_dir in tqdm(os.listdir(logdir)):
        if run_dir.startswith('.'):
            # Ignore hidden files.
            continue

        if not os.path.isdir(os.path.join(logdir, run_dir)):
            # Ignore other things such as generated tables.
            print(run_dir)
            continue

        count += 1

        config_dir = os.path.join(logdir, run_dir, 'config.yaml')
        with open(config_dir) as file:
            config = yaml.full_load(file)
        config = squeeze_dict(config)

        current_run = {'run': run_dir}
        for var in variables:
            # record required variables in config.yaml.
            current_run[var] = config[var]

        # for metric in ['test_mrr', 'test_rck1', 'test_rck3', 'test_rck10', 'test_loss',
        #                'val_mrr', 'val_rck1', 'val_rck3', 'val_rck10', 'val_loss']:
        for metric in ['test_mrr']:
            event_path = os.path.join(logdir, run_dir, metric)
            # print(f'Processing event file {event_path}')

            ea = EventAccumulator(event_path).Reload()

            tag_values = []
            steps = []

            x = 'test' if metric.startswith('test') else 'val'
            for event in ea.Scalars(x):
                # Each (value, step) corresponds to a (value, snapshot).
                tag_values.append(event.value)
                steps.append(event.step)

            current_run['average_' + metric] = np.mean(tag_values)
        # current_run: one row in the aggregated dataset.
        all_runs.append(current_run)
    print(f'exported {count} experiments.')
    return pd.DataFrame(all_runs)


def collapse_over_seed(x):
    x['mrr'] = x['average_test_mrr'].mean()
    return x


def gather_essential_fields(df):
    df['seed'] = df.run.apply(lambda x: x[-1])
    df['run'] = df.run.apply(lambda x: x[:-2])
    df['model'] = df['gnn.layer_type'] + ':' + df['gnn.embed_update_method']
    df['data'] = df['dataset.name']
    return df


if __name__ == '__main__':
    variables = ['dataset.format', 'dataset.name',
                 'gnn.layer_type', 'gnn.batchnorm', 'gnn.layers_mp',
                 'gnn.skip_connection', 'gnn.embed_update_method',
                 'optim.base_lr',
                 'transaction.feature_int_dim',
                 'gnn.agg', 'train.mode',
                 'train.internal_validation_tolerance', 'gnn.dim_inner',
                 'meta.is_meta', 'meta.method', 'meta.alpha',
                 'transaction.snapshot_freq', 'gnn.embed_update_method']

    dataset_columns = ['bsi_zk_2008.tsv', 'AS-733', 'reddit-title.tsv', 'reddit-body.tsv',  'bsi_svt_2008.tsv', 'CollegeMsg.txt', 'bitcoinotc.csv', 'bitcoinalpha.csv']

    for table_name in ['Table2', 'Table3_top', 'Table3_middle', 'Table3_bottom_and_Table4']:
        df = tabulate_events(table_name, variables)
        print('=' * 200)
        print(table_name)
        print('=' * 200)

        df = gather_essential_fields(df)
        df = df.groupby('run').apply(collapse_over_seed).sort_values(by='mrr', ascending=True)
        table = df.groupby(['data', 'model']).last().reset_index().pivot(index='model', columns='data', values='mrr')

        print(table[[c for c in dataset_columns if c in table.columns]].to_markdown())
        if table_name == 'Table3_bottom_and_Table4':
            best_alpha = df.groupby(['data', 'model']).last()['meta.alpha'].reset_index().pivot(index='model', columns='data', values='meta.alpha')
            print(best_alpha[dataset_columns].to_markdown())
            table_non_meta = df[df['meta.alpha'] == 1].groupby(['data', 'model']).last().reset_index().pivot(index='model', columns='data', values='mrr')
            print(table_non_meta[dataset_columns].to_markdown())
            gain = (table / table_non_meta - 1) * 100
            gain = gain[dataset_columns]
            gain['Average'] = gain.mean(axis=1)
            print(gain.to_markdown())
