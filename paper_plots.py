import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


results_0 = pd.read_csv('run-article_results_long_0-tag-Train_reward.csv')
results_1 = pd.read_csv('run-article_results_long_1-tag-Train_reward.csv')
results_2 = pd.read_csv('run-article_results_long_2-tag-Train_reward.csv')

min_size = 100000
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
plt.savefig("results.png")
