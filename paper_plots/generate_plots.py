import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_short = [
    'paper_plots/run-article_results_short_1-tag-Train_reward.csv',
    'paper_plots/run-article_results_short_2-tag-Train_reward.csv',
    'paper_plots/run-article_results_short_3-tag-Train_reward.csv',
    'paper_plots/short_train_reward.png',
    'gail'
]
results_short_ct = [
    'paper_plots/run-article_results_short_ct_1-tag-Train_reward.csv',
    'paper_plots/run-article_results_short_ct_2-tag-Train_reward.csv',
    'paper_plots/run-article_results_short_ct_3-tag-Train_reward.csv',
    'paper_plots/short_ct_train_reward.png',
    'gail + dropout'
]
results_short_bc_ct = [
    'paper_plots/run-article_results_short_ct_bc_1-tag-Train_reward.csv',
    'paper_plots/run-article_results_short_ct_bc_2-tag-Train_reward.csv',
    'paper_plots/run-article_results_short_ct_bc_3-tag-Train_reward.csv',
    'paper_plots/short_bc_ct_train_reward.png',
    'bc + gail + dropout'
]

for results_files in [results_short, results_short_ct, results_short_bc_ct]:
    results_0 = pd.read_csv(results_files[0])
    results_1 = pd.read_csv(results_files[1])
    results_2 = pd.read_csv(results_files[2])

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
    results_id = np.arange(results.shape[1])

    plt.plot(results_id, results_mean)
    plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)
    plt.grid()
    plt.savefig(results_files[3])
    plt.clf()

for results_files in [results_short, results_short_ct, results_short_bc_ct]:
    results_0 = pd.read_csv(results_files[0])
    results_1 = pd.read_csv(results_files[1])
    results_2 = pd.read_csv(results_files[2])

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

    plt.plot(results_id, results_mean, label=results_files[4])
    plt.fill_between(results_id, results_mean + results_std, results_mean - results_std, alpha=0.5)

plt.xlabel(r'# environment interactions ($ \times 10^5$)')
plt.ylabel('Reward')
plt.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.savefig('paper_plots/short_all.eps')

# generate eps