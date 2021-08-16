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
rcParams["savefig.dpi"] = 150

results_dataset = [
    [
        'paper_plots/long/run-article_results_long_1-tag-dis_loss.csv',
        'paper_plots/long/run-article_results_long_1-tag-disc_pre_loss.csv',
        'gail'
    ],
    [
        'paper_plots/long_ct/run-article_results_long_ct_1-tag-dis_loss.csv',
        'paper_plots/long_ct/run-article_results_long_ct_1-tag-disc_pre_loss.csv',
        'gail + dropout'
    ],
    [
        'paper_plots/long_ct_bc/run-article_results_long_ct_bc_1-tag-dis_loss.csv',
        'paper_plots/long_ct_bc/run-article_results_long_ct_bc_1-tag-disc_pre_loss.csv',
        'bc + gail + dropout'
    ],
    # [
    #     'paper_plots/long_bc/run-article_results_long_bc_1-tag-dis_loss.csv',
    #     'paper_plots/long_bc/run-article_results_long_bc_1-tag-disc_pre_loss.csv',
    #     'bc + gail'
    # ]
]

for results_files in results_dataset:
    dis_loss_list = []
    dis_loss_list.append(pd.read_csv(results_files[0]))

    dis_val_loss_list = []
    dis_val_loss_list.append(pd.read_csv(results_files[1]))

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
    results_id = np.arange(results.shape[1]) * 7200 /100000

    plt.plot(results_id, results_mean, label=results_files[2])
    plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

plt.xlabel(r'environment interactions ($ \times 10^5$)')
plt.ylabel('Reward')
plt.legend(loc='upper left', shadow=True)
plt.savefig('paper_plots/plots/long_dis_val_dif.png')
plt.clf()
