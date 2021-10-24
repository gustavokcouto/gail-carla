import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from pathlib import Path


fontsize = 13
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.size'] = fontsize
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.titlesize'] = fontsize
rcParams['axes.labelsize'] = fontsize
rcParams['text.usetex'] = True
rcParams["savefig.dpi"] = 150

fig_all, ax_all = plt.subplots(3)


def generate_vector_field(vector_field, title, ax_count):
    vector_field = vector_field.reset_index()
    x_min = vector_field['x'].min()
    x_max = vector_field['x'].max()
    vector_field['x'] = (vector_field['x'] - x_min) / (x_max - x_min)
    y_min = vector_field['y'].min()
    y_max = vector_field['y'].max()
    vector_field['y'] = (vector_field['y'] - y_min) / (y_max - y_min)
    vel_x_max = vector_field['velocity_x'].max()
    vel_y_max = vector_field['velocity_y'].max()
    vector_field['velocity_x'] = 0.01 * vector_field['velocity_x'] / \
        (vel_x_max ** 2 + vel_y_max ** 2) ** 0.5
    vector_field['velocity_y'] = 0.01 * vector_field['velocity_y'] / \
        (vel_x_max ** 2 + vel_y_max ** 2) ** 0.5
    vector_field['not_episode_end'] = vector_field['episode'].isna()
    vector_field['episode_end'] = vector_field['not_episode_end'].apply(
        lambda not_episode_end: not not_episode_end)
    vector_field['color'] = vector_field['episode_end'].apply(
        lambda episode_end: 'red' if episode_end else 'yellow')
    vector_field.to_csv('train_episodes.csv')
    vector_field_episode_end = vector_field[vector_field['episode_end'] == True]
    vector_field = vector_field.iloc[::20, :]
    vector_field = vector_field.append(
        vector_field_episode_end, ignore_index=True)
    ax_all[ax_count].quiver(
        vector_field['x'],
        1 - vector_field['y'],
        vector_field['velocity_x'],
        -1 * vector_field['velocity_y'],
        color=vector_field['color'],
        scale=1)
    ax_all[ax_count].xaxis.set_ticks([])
    ax_all[ax_count].yaxis.set_ticks([])
    ax_all[ax_count].axis([-0.1, 1.1, -0.1, 1.1])
    max_ep_count = vector_field['ep_count'].max()
    min_ep_count = vector_field['ep_count'].min() - 1
    timesteps_per_episode = 7200
    ax_all[ax_count].set_title(title + r' ({}-{} $ \times 10^3$ environment interactions)'.format(
        int(min_ep_count * timesteps_per_episode / 1000), int(max_ep_count * timesteps_per_episode / 1000)))


ax_count = 0
rcParams["savefig.dpi"] = 300
img_path = Path('paper_plots/vector_field/output')
title = 'Early train'
train_vector_field = pd.read_csv(
    'article_results/long_bc/1/env_info/train_env_0.csv')
episode_mean_len = 2000
vector_field = train_vector_field.iloc[0:0 + 5 * episode_mean_len]
generate_vector_field(vector_field, title, ax_count)
ax_count += 1

vector_field = train_vector_field[(train_vector_field['ep_count'] > 120) & (
    train_vector_field['ep_count'] < 160)]
img_name = 'middle_train'
title = 'Middle train'
generate_vector_field(vector_field, title, ax_count)
ax_count += 1

vector_field = train_vector_field[(train_vector_field['ep_count'] > 260)]
img_name = 'late_train'
title = 'Late train'
generate_vector_field(vector_field, title, ax_count)

# img_name = 'early_eval'
# eval_vector_field = pd.read_csv('article_results/long_bc/1/env_info/eval_env.csv')
# episode_mean_len = 2000
# vector_field = eval_vector_field[(eval_vector_field['ep_count'] < 20)]
# generate_vector_field(vector_field, img_name, img_path)

# vector_field = eval_vector_field[(eval_vector_field['ep_count'] > 30)&(eval_vector_field['ep_count'] < 50)]
# img_name = 'middle_eval'
# generate_vector_field(vector_field, img_name, img_path)

# vector_field = eval_vector_field[(eval_vector_field['ep_count'] > 65)]
# img_name = 'late_eval'
# generate_vector_field(vector_field, img_name, img_path)

# expert_vector_field = pd.read_csv('gail_experts/route_00/expert.csv')
# vector_field = expert_vector_field[expert_vector_field['ep_count'] == 1]
# img_name = 'expert'
# generate_vector_field(vector_field, img_name, img_path)
plt.subplots_adjust(hspace=0.3)
fig_all.savefig(img_path / 'train_all.png', bbox_inches='tight')
fig_all.savefig(img_path / 'train_all.eps', bbox_inches='tight')
