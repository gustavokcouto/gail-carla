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
        'paper_plots/long/run-article_results_long_1-tag-Train_reward.csv',
        'paper_plots/long/run-article_results_long_2-tag-Train_reward.csv',
        'paper_plots/long/run-article_results_long_1-tag-Eval_reward.csv',
        'paper_plots/long/run-article_results_long_2-tag-Eval_reward.csv',
        'gail'
    ],
    [
        'paper_plots/long_bc/run-article_results_long_bc_1-tag-Train_reward.csv',
        'paper_plots/long_bc/run-article_results_long_bc_4-tag-Train_reward.csv',
        'paper_plots/long_bc/run-article_results_long_bc_1-tag-Eval_reward.csv',
        'paper_plots/long_bc/run-article_results_long_bc_4-tag-Eval_reward.csv',
        'bc\_gail'
    ]
]

for results_files in results_dataset:
    results_0 = pd.read_csv(results_files[0])
    results_1 = pd.read_csv(results_files[1])

    min_size = 420
    results_list = [results_0, results_1]
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
    results_id = np.arange(results.shape[1]) * 7200 /100000

    plt.plot(results_id, results_mean, label=results_files[4])
    plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

results_mean.fill(173.6)
plt.plot(results_id, results_mean, label='bc')
results_std.fill(137.35516007780703)
plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

plt.xlabel(r'environment interactions ($ \times 10^5$)')
plt.ylabel('Reward')
plt.legend(loc='lower right', shadow=True, fontsize='medium')
plt.savefig('paper_plots/plots/long_train_reward.png')
plt.clf()

for results_files in results_dataset:
    results_0 = pd.read_csv(results_files[2])
    results_1 = pd.read_csv(results_files[3])

    min_size = 420
    results_list = [results_0, results_1]
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
    results_id = np.arange(results.shape[1]) * 7200 /100000

    plt.plot(results_id, results_mean, label=results_files[4])
    plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

plt.xlabel(r'environment interactions ($ \times 10^5$)')
plt.ylabel('Reward')
plt.savefig('paper_plots/plots/long_eval_reward.png')
