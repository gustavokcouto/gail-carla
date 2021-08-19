import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from pathlib import Path


def generate_vector_field(vector_field, img_name, img_path):
    vector_field = vector_field.reset_index()
    x_min = vector_field['x'].min()
    x_max = vector_field['x'].max()
    vector_field['x'] = (vector_field['x'] - x_min) / (x_max - x_min)
    y_min = vector_field['y'].min()
    y_max = vector_field['y'].max()
    vector_field['y'] = (vector_field['y'] - y_min) / (y_max - y_min)
    vel_x_max = vector_field['velocity_x'].max()
    vel_y_max = vector_field['velocity_y'].max()
    vector_field['velocity_x'] = 0.01 * vector_field['velocity_x'] / (vel_x_max ** 2 + vel_y_max ** 2) ** 0.5
    vector_field['velocity_y'] = 0.01 * vector_field['velocity_y'] / (vel_x_max ** 2 + vel_y_max ** 2) ** 0.5
    vector_field['not_episode_end'] = vector_field['episode'].isna()
    vector_field['episode_end'] = vector_field['not_episode_end'].apply(lambda not_episode_end: not not_episode_end)
    vector_field['color'] = vector_field['episode_end'].apply(lambda episode_end: 'red' if episode_end else 'yellow')
    vector_field.to_csv('train_episodes.csv')
    vector_field_episode_end = vector_field[vector_field['episode_end'] == True]
    vector_field = vector_field.iloc[::20, :]
    vector_field = vector_field.append(vector_field_episode_end, ignore_index=True)
    fig, ax = plt.subplots()
    ax.quiver(
        vector_field['x'],
        1 - vector_field['y'],
        vector_field['velocity_x'],
        -1 * vector_field['velocity_y'],
        color=vector_field['color'],
        scale=1)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    filename = img_name + '.png'
    fig.savefig(img_path / filename)
    filename = img_name + '.eps'
    fig.savefig(img_path / filename)

rcParams["savefig.dpi"] = 300
img_path = Path('paper_plots/vector_field/output')
img_name = 'early_train'
train_vector_field = pd.read_csv('article_results/long_bc/1/env_info/train_env_0.csv')
episode_mean_len = 2000
vector_field = train_vector_field.iloc[0:0 + 20 * episode_mean_len]
generate_vector_field(vector_field, img_name, img_path)

vector_field = train_vector_field[(train_vector_field['ep_count'] > 120)&(train_vector_field['ep_count'] < 160)]
img_name = 'middle_train'
generate_vector_field(vector_field, img_name, img_path)

vector_field = train_vector_field[(train_vector_field['ep_count'] > 260)]
img_name = 'late_train'
generate_vector_field(vector_field, img_name, img_path)

img_name = 'early_eval'
eval_vector_field = pd.read_csv('article_results/long_bc/1/env_info/eval_env.csv')
episode_mean_len = 2000
vector_field = eval_vector_field[(eval_vector_field['ep_count'] < 20)]
generate_vector_field(vector_field, img_name, img_path)

vector_field = eval_vector_field[(eval_vector_field['ep_count'] > 30)&(eval_vector_field['ep_count'] < 50)]
img_name = 'middle_eval'
generate_vector_field(vector_field, img_name, img_path)

vector_field = eval_vector_field[(eval_vector_field['ep_count'] > 65)]
img_name = 'late_eval'
generate_vector_field(vector_field, img_name, img_path)

expert_vector_field = pd.read_csv('gail_experts/route_00/expert.csv')
vector_field = expert_vector_field[expert_vector_field['ep_count'] == 1]
img_name = 'expert'
generate_vector_field(vector_field, img_name, img_path)