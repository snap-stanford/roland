"""
Use this script to have a fast and clean comparison of different configuration
in the grid search.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='ticks', context='poster')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
np.set_printoptions(precision=3, linewidth=200, suppress=True)


# Configure plots here.
control_variables = ['l_pre', 'l_mp', 'l_post', 'dim_gnn', 'skip', 'lr']
objective_variables = ['loss', 'auc', 'mrr', 'rck1', 'rck3', 'rck10']


def plot_analysis(fname, division='test', dataset=None, metric='accuracy',
                  rank_resolution=0.001, filter=None, filter_rm=None):
    results_file_path = '../run/results/{}/agg/{}.csv'.format(fname, division)
    df = pd.read_csv(results_file_path)
    df = df.fillna(0)
    df['epoch'] += 1
    ncols = len(control_variables)
    nrows = len(objective_variables)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=125,
                             figsize=(5*nrows, 5*ncols))

    for i, obj in enumerate(objective_variables):
        for j, var in enumerate(control_variables):
            sns.violinplot(x=var, y=obj, data=df, ax=axes[i, j])
    plt.tight_layout()
    fig.savefig('./figs/{}.png'.format(results_file_path.replace('/', '_')))


if __name__ == '__main__':
    experiment_name = 'manually_added'
    plot_analysis(experiment_name, division='val', dataset='all',
                  metric='accuracy', rank_resolution=0)
