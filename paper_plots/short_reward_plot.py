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
        'paper_plots/short/run-article_results_short_1-tag-Train_reward.csv',
        'paper_plots/short/run-article_results_short_2-tag-Train_reward.csv',
        'paper_plots/short/run-article_results_short_3-tag-Train_reward.csv',
        'paper_plots/short/run-article_results_short_1-tag-Eval_reward.csv',
        'paper_plots/short/run-article_results_short_2-tag-Eval_reward.csv',
        'paper_plots/short/run-article_results_short_3-tag-Eval_reward.csv',
        'gail'
    ],
    # [
    #     'paper_plots/short_ct/run-article_results_short_ct_1-tag-Train_reward.csv',
    #     'paper_plots/short_ct/run-article_results_short_ct_2-tag-Train_reward.csv',
    #     'paper_plots/short_ct/run-article_results_short_ct_3-tag-Train_reward.csv',
    #     'paper_plots/short_ct/run-article_results_short_ct_1-tag-Eval_reward.csv',
    #     'paper_plots/short_ct/run-article_results_short_ct_2-tag-Eval_reward.csv',
    #     'paper_plots/short_ct/run-article_results_short_ct_3-tag-Eval_reward.csv',
    #     'gail + dropout'
    # ],
    # [
    #     'paper_plots/short_ct_bc/run-article_results_short_ct_bc_1-tag-Train_reward.csv',
    #     'paper_plots/short_ct_bc/run-article_results_short_ct_bc_2-tag-Train_reward.csv',
    #     'paper_plots/short_ct_bc/run-article_results_short_ct_bc_3-tag-Train_reward.csv',
    #     'paper_plots/short_ct_bc/run-article_results_short_ct_bc_1-tag-Eval_reward.csv',
    #     'paper_plots/short_ct_bc/run-article_results_short_ct_bc_2-tag-Eval_reward.csv',
    #     'paper_plots/short_ct_bc/run-article_results_short_ct_bc_3-tag-Eval_reward.csv',
    #     'bc + gail + dropout'
    # ],
    [
        'paper_plots/short_bc/run-article_results_short_bc_1-tag-Train_reward.csv',
        'paper_plots/short_bc/run-article_results_short_bc_2-tag-Train_reward.csv',
        'paper_plots/short_bc/run-article_results_short_bc_3-tag-Train_reward.csv',
        'paper_plots/short_bc/run-article_results_short_bc_1-tag-Eval_reward.csv',
        'paper_plots/short_bc/run-article_results_short_bc_2-tag-Eval_reward.csv',
        'paper_plots/short_bc/run-article_results_short_bc_3-tag-Eval_reward.csv',
        'bc\_gail'
    ]
]

for results_files in results_dataset:
    results_0 = pd.read_csv(results_files[0])
    results_1 = pd.read_csv(results_files[1])
    results_2 = pd.read_csv(results_files[2])

    min_size = 350
    results_list = [results_0, results_1, results_2]
    for results in results_list:
        min_size = min(min_size, len(results))

    results_arr_list = []
    for results in results_list:
        results_arr = results['Value'].to_numpy()
        results_arr = results_arr[:min_size]
        results_arr_list.append(results_arr)

    results = np.array(results_arr_list)
    # results = results.reshape(-1, 3)
    results_mean = results.mean(axis=0)
    results_std = results.std(axis=0)
    results_id = np.arange(results.shape[1]) * 2000 /100000

    plt.plot(results_id, results_mean, label=results_files[6])
    plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

plt.xlabel(r'environment interactions ($ \times 10^5$)')
plt.ylabel('Reward')
plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('paper_plots/plots/short_train_reward.png', dpi=150)
plt.clf()

for results_files in results_dataset:
    results_0 = pd.read_csv(results_files[3])
    results_1 = pd.read_csv(results_files[4])
    results_2 = pd.read_csv(results_files[5])

    min_size = 350000
    results_list = [results_0, results_1, results_2]
    for results in results_list:
        min_size = min(min_size, len(results))

    results_arr_list = []
    for results in results_list:
        results_arr = results['Value'].to_numpy()
        results_arr = results_arr[:min_size]
        results_arr_list.append(results_arr)

    results = np.array(results_arr_list)
    # results = results.reshape(-1, 3)
    results_mean = results.mean(axis=0)
    results_std = results.std(axis=0)
    results_id = np.arange(results.shape[1]) * 2000 /100000

    plt.plot(results_id, results_mean, label=results_files[6])
    plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

plt.xlabel(r'environment interactions ($ \times 10^5$)')
plt.ylabel('Reward')
plt.savefig('paper_plots/plots/short_eval_reward.png', dpi=150)
