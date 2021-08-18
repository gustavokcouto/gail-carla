import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams["savefig.dpi"] = 150
train_vector_field = pd.read_csv('paper_plots/env_info/eval_env.csv')
steps_per_epoch = 720
print(train_vector_field.shape)
print(train_vector_field.shape)
x_min = train_vector_field['x'].min()
x_max = train_vector_field['x'].max()
train_vector_field['x'] = (train_vector_field['x'] - x_min) / (x_max - x_min)
y_min = train_vector_field['y'].min()
y_max = train_vector_field['y'].max()
train_vector_field['y'] = (train_vector_field['y'] - y_min) / (y_max - y_min)
vel_x_max = train_vector_field['velocity_x'].max()
vel_y_max = train_vector_field['velocity_y'].max()
train_vector_field['velocity_x'] = 0.01 * train_vector_field['velocity_x'] / (vel_x_max ** 2 + vel_y_max ** 2) ** 0.5
train_vector_field['velocity_y'] = 0.01 * train_vector_field['velocity_y'] / (vel_x_max ** 2 + vel_y_max ** 2) ** 0.5
train_vector_field['color'] = train_vector_field['episode'].isna()
train_vector_field = train_vector_field[(train_vector_field['ep_count'] < 10) & (train_vector_field['ep_count'] > 5)]
train_vector_field_end_episode = train_vector_field[(train_vector_field['color'] == False)]
train_vector_field = train_vector_field.iloc[::20, :]
train_vector_field = train_vector_field.append(train_vector_field_end_episode, ignore_index=True)
train_vector_field['color'] = train_vector_field['color'].apply(lambda color: 'red' if color else 'yellow')
print(train_vector_field['color'])
# train_vector_field = train_vector_field.sample(50)
# train_vector_field = train_vector_field.sort_values(['x', 'y'])
# plt.streamplot(train_vector_field['x'], train_vector_field['y'], train_vector_field['velocity_x'], train_vector_field['velocity_y'], density=1.4, linewidth=None, color='#A23BEC')
fig, ax = plt.subplots()
ax.quiver(
    train_vector_field['x'],
    1 - train_vector_field['y'],
    train_vector_field['velocity_x'],
    -1 * train_vector_field['velocity_y'],
    color=train_vector_field['color'],
    scale=1)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.axis([-0.2, 1.2, -0.2, 1.2])
fig.savefig('paper_plots/vector_field/early_eval.png')
fig.savefig('paper_plots/vector_field/early_eval.eps')