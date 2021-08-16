import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


fontsize = 13
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.size'] = fontsize
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.titlesize'] = fontsize
rcParams['axes.labelsize'] = fontsize
rcParams['text.usetex'] = True

results_dataset = [
    [
        'paper_plots/short/run-article_results_short_1-tag-dis_loss.csv',
        'paper_plots/short/run-article_results_short_2-tag-dis_loss.csv',
        'paper_plots/short/run-article_results_short_3-tag-dis_loss.csv',
        'paper_plots/short/run-article_results_short_1-tag-disc_pre_loss.csv',
        'paper_plots/short/run-article_results_short_2-tag-disc_pre_loss.csv',
        'paper_plots/short/run-article_results_short_3-tag-disc_pre_loss.csv',
        'gail'
    ],
    [
        'paper_plots/short_ct/run-article_results_short_ct_1-tag-dis_loss.csv',
        'paper_plots/short_ct/run-article_results_short_ct_2-tag-dis_loss.csv',
        'paper_plots/short_ct/run-article_results_short_ct_3-tag-dis_loss.csv',
        'paper_plots/short_ct/run-article_results_short_ct_1-tag-disc_pre_loss.csv',
        'paper_plots/short_ct/run-article_results_short_ct_2-tag-disc_pre_loss.csv',
        'paper_plots/short_ct/run-article_results_short_ct_3-tag-disc_pre_loss.csv',
        'gail + dropout'
    ],
    [
        'paper_plots/short_ct_bc/run-article_results_short_ct_bc_1-tag-dis_loss.csv',
        'paper_plots/short_ct_bc/run-article_results_short_ct_bc_2-tag-dis_loss.csv',
        'paper_plots/short_ct_bc/run-article_results_short_ct_bc_3-tag-dis_loss.csv',
        'paper_plots/short_ct_bc/run-article_results_short_ct_bc_1-tag-disc_pre_loss.csv',
        'paper_plots/short_ct_bc/run-article_results_short_ct_bc_2-tag-disc_pre_loss.csv',
        'paper_plots/short_ct_bc/run-article_results_short_ct_bc_3-tag-disc_pre_loss.csv',
        'bc + gail + dropout'
    ],
    [
        'paper_plots/short_bc/run-article_results_short_bc_1-tag-dis_loss.csv',
        'paper_plots/short_bc/run-article_results_short_bc_2-tag-dis_loss.csv',
        'paper_plots/short_bc/run-article_results_short_bc_3-tag-dis_loss.csv',
        'paper_plots/short_bc/run-article_results_short_bc_1-tag-disc_pre_loss.csv',
        'paper_plots/short_bc/run-article_results_short_bc_2-tag-disc_pre_loss.csv',
        'paper_plots/short_bc/run-article_results_short_bc_3-tag-disc_pre_loss.csv',
        'bc + gail'
    ]
]

for results_files in results_dataset:
    dis_loss_list = []
    dis_loss_list.append(pd.read_csv(results_files[0]))
    dis_loss_list.append(pd.read_csv(results_files[1]))
    dis_loss_list.append(pd.read_csv(results_files[2]))

    dis_val_loss_list = []
    dis_val_loss_list.append(pd.read_csv(results_files[3]))
    dis_val_loss_list.append(pd.read_csv(results_files[4]))
    dis_val_loss_list.append(pd.read_csv(results_files[5]))

    min_size = 350
    for results in dis_loss_list:
        min_size = min(min_size, len(results))

    results_arr_list = []
    for dis_loss, dis_val_loss in zip(dis_loss_list, dis_val_loss_list):
        results_arr = dis_loss['Value'].to_numpy() - dis_val_loss['Value'].to_numpy()
        results_arr = results_arr[:min_size]
        results_arr_list.append(results_arr)

    results = np.array(results_arr_list)
    results_mean = results.mean(axis=0)
    results_std = results.std(axis=0)
    results_id = np.arange(results.shape[1]) * 2000 /100000

    plt.plot(results_id, results_mean, label=results_files[6])
    plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

plt.xlabel(r'environment interactions ($ \times 10^5$)')
plt.ylabel('Reward')
plt.legend(loc='upper left', shadow=True, fontsize='medium')
plt.savefig('paper_plots/plots/short_dis_val_dif.eps')
plt.savefig('paper_plots/plots/short_dis_val_dif.png')
plt.clf()
