import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams["savefig.dpi"] = 150
expert_vector_field = pd.read_csv('gail_experts/route_00/expert.csv')
train_vector_field = pd.read_csv('paper_plots/env_info/train_env_0.csv')
expert_vector_field['color'] = 'yellow'
expert_vector_field = expert_vector_field[expert_vector_field['ep_count'] == 1]
# train_vector_field = train_vector_field[(train_vector_field['ep_count'] < 240) & (train_vector_field['ep_count'] > 200)]
# train_vector_index = train_vector_field[train_vector_field['ep_count'] == 200].index.min()
train_vector_field = train_vector_field.iloc[0:0 + 20 * expert_vector_field.shape[0]]
# train_vector_field = train_vector_field.sample(expert_vector_field.shape[0])
train_vector_field['color'] = 'red'
vector_field = train_vector_field.append(expert_vector_field, ignore_index=True)
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
vector_field = vector_field.iloc[::20, :]
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
fig.savefig('paper_plots/vector_field/train_expert.png')
fig.savefig('paper_plots/vector_field/train_expert.eps')