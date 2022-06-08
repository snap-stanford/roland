"""
Download tensorboard from TensorBoard Dev and summarize
results to a table.

Author: Tianyu Du
Date: Jun. 1, 2022
"""
import sys
import os
import shutil

import numpy as np
import tensorboard as tb

def get_model(x):
    for z in x.split('-'):
        if 'model=' in z:
            return z.replace('model=', '')

def get_update(x):
    for z in x.split('-'):
        if 'update=' in z:
            return z.replace('update=', '')

def get_data(x):
    for z in x.split('-'):
        if 'name=' in z:
            return z.replace('name=', '')

def get_alpha(x):
    for z in x.split('-'):
        if 'alpha=' in z:
            return z.replace('alpha=', '')

get_seed = lambda x: x.split('\\')[-1].split('/')[0]

if __name__ == '__main__':
    experiment_id = sys.argv[1]
    # TODO: remove this, for debugging purpose.
    experiment_id = 'nRmHLIBrTri2NgpR2WE28A'
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    df_raw = df.copy()
    df = df_raw.copy()

    table = 'table_3_top'

    if table == 'table_2':
        df = df[df.run.str.endswith('test_mrr')]
        df = df.groupby('run')['value'].mean().reset_index()
        df['data'] = df.run.apply(lambda x: x.split('-name=')[1].split('-freq=')[0])
        df['model'] = df.run.apply(lambda x: x.split('-model=')[1].split('\\')[0])

        df.groupby(['data', 'model']).mean().reset_index().pivot(index='model', columns='data', values='value')

        # df = df[df['tag'] == 'test']
        # df = df[df.run.str.endswith('test_mrr')]

        # path_ours = '/home/tianyudu/Development/GraphGym_dev/run/temp/runs_fixed_split_v19'
        # df = tabulate_events(path_ours, variables)
        # df['seed'] = df.run.apply(lambda x: x[-1])
        # df['run'] = df['run'].apply(lambda x: x[:-2])
        # df = df.groupby('run')['average_test_mrr'].mean().reset_index()
        # df['data'] = df.run.apply(lambda x: [z for z in x.split('-') if z.startswith('name=')][0].replace('name=', ''))
        # df['model'] = df.run.apply(lambda x: [z for z in x.split('-') if z.startswith('update=')][0].replace('update=', '').split('/')[0])

        # df.groupby(['data', 'model'])['average_test_mrr'].max().reset_index().pivot(index='model', columns='data', values='average_test_mrr')
        # df = df.sort_values(by='average_test_mrr', ascending=False)

        # df.groupby(['data', 'model']).first().to_csv('./temp.csv')

    if table.startswith('table_3'):
        df = df[df.run.str.endswith('test_mrr')]
        if table == 'table_3_top':
            pass
        elif table == 'table_3_middle':
            df['data'] = df.run.apply(lambda x: x.split('-name=')[-1].split('-freq')[0])
            df['model'] = df.run.apply(get_model).fillna('TGCN').apply(lambda x: x.split('\\')[0])
            df['seed'] = df.run.apply(lambda x: x.split('\\')[-1].split('/')[0])
        elif table == 'table_3_bottom':
            df['data'] = df.run.apply(lambda x: x.split('\\')[1].split('_')[0])
            df['model'] = df.run.apply(lambda x: x.split('\\')[1].split('_')[1])
            df['seed'] = df.run.apply(lambda x: x.split('\\')[-1].split('/')[0])

        df = df.groupby(['data', 'model', 'seed'])['value'].mean()

        mean = df.groupby(['data', 'model']).mean().reset_index().pivot(index='model', columns='data')
        std = (df.groupby(['data', 'model']).std() / np.sqrt(3)).reset_index().pivot(index='model', columns='data')

        df_repr = mean.round(3).astype(str) + ' \pm ' + std.round(3).astype(str)


    elif table.startswith('table_4'):
        path_ours = '/home/tianyudu/Development/GraphGym_dev/run/temp/runs_all_v2'
        df = tabulate_events(path_ours, variables)
        df['dir'] = df['run'].copy()
        df['seed'] = df.run.apply(lambda x: x[-1])
        df['run'] = df['run'].apply(lambda x: x[:-2])

        # get best non-meta results.
        df = df[df['meta.alpha'] == 1]

        def f(x):
            x['mrr'] = x['average_test_mrr'].mean()
            return x

        df = df.groupby('run').apply(f)

        df = df.sort_values(by='mrr', ascending=False)
        best = df.groupby(['dataset.name', 'gnn.embed_update_method']).first().reset_index()

        best.pivot(index='gnn.embed_update_method', columns='dataset.name', values='meta.alpha')
        best.pivot(index='gnn.embed_update_method', columns='dataset.name', values='mrr')

        # pull out configs of best.
        for i in range(len(best)):
            data = best['dataset.name'].iloc[i]
            model = best['gnn.embed_update_method'].iloc[i]
            alpha = best['meta.alpha'].iloc[i]

            config_file_name = data + '_' + model + '_' + str(alpha) + '.yaml'
            out_path = '/home/tianyudu/Development/GraphGym_dev/run/replications/table4'
            dest = os.path.join(out_path, config_file_name)

            src = os.path.join(path_ours, best['dir'].iloc[i], 'config.yaml')
            shutil.copyfile(src, dest)

        df.groupby(['data', 'model'])['average_test_mrr'].max().reset_index().pivot(index='model', columns='data', values='average_test_mrr')



        df = df[df.run.str.endswith('test_mrr')]
        df = df.groupby('run')['value'].mean().reset_index()
        df['data'] = df.run.apply(lambda x: x.split('\\')[1].split('_')[0])
        df['model'] = df.run.apply(lambda x: x.split('\\')[1].split('_')[-2])
        with_meta = df.pivot(index='model', columns='data', values='value').reindex(['average', 'mlp', 'gru'])[
            ['reddit-title.tsv', 'reddit-body.tsv', 'bsi', 'CollegeMsg.txt', 'bitcoinotc.csv', 'bitcoinalpha.csv']
        ]

        without_meta = np.array([
            [0.362, 0.289, 0.177, 0.075, 0.120, 0.0962],
            [0.395, 0.291, 0.217, 0.103, 0.154, 0.148],
            [0.425, 0.362, 0.205, 0.112, 0.194, 0.157]
        ])

        (with_meta / without_meta * 100 - 100)


    # TODO: might need to make this easier?
    # from single run results.
    df['data'] = df.run.apply(lambda x: x.split('\\')[1].split('_')[0])
    df['model'] = df.run.apply(lambda x: x.split('\\')[1].split('_')[1])
    df['seed'] = df.run.apply(lambda x: x.split('\\')[-1].split('/')[0])

    # from batch run results.

    df['model'] = df.run.apply(get_model)
    df['seed'] = df.run.apply(get_seed)

    # for table 4.
    df['data'] = df.run.apply(lambda x: x.split('=')[0].strip('_alpha').strip('results\\').split('_')[0])
    df['model'] = df.run.apply(lambda x: x.split('=')[0].strip('_alpha').strip('results\\').split('_')[1])
    df['seed'] = df.run.apply(lambda x: x.split('/')[0][-1])
    df['alpha'] = df.run.apply(get_alpha)

    # average over snapshots.

    df_repr.to_csv('./tb_output.csv')
    print(df_repr.to_latex())

    # ==== for meta learning.
    df = df.groupby(['data', 'model', 'seed', 'alpha'])['value'].mean()

    mean = df.groupby(['data', 'model', 'alpha']).mean()
    std = df.groupby(['data', 'model', 'alpha']).std() / np.sqrt(3)

    mean = mean.reset_index().pivot(index='model', columns='data')
    std = std.reset_index().pivot(index='model', columns='data')

    df_repr = mean.round(3).astype(str) + ' \pm ' + std.round(3).astype(str)

    df_repr.to_csv('./tb_output.csv')
    print(df_repr.to_latex())