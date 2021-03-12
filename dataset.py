from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np


DATASET_DIR = Path('data/carla')


def save_dataset(file_path):
    states = []
    metrics = []
    actions = []
    rewards = []
    lens = []

    episode_paths = sorted((DATASET_DIR).glob('*'))
    for episode_path in episode_paths:
        states_ep = []
        metrics_ep = []
        actions_ep = []
        rewards_ep = []
        measurements = pd.read_csv(episode_path/'measurements.csv')
        for step, image_path in enumerate(sorted((episode_path/'rgb').glob('*.png'))):
            rgb = Image.open(image_path)
            rgb = np.array(rgb)
            rgb = np.transpose(rgb, (2, 0, 1))
            steer = measurements.iloc[step]['steer']
            throttle = measurements.iloc[step]['throttle']
            actions_ep.append([steer, throttle])
            compass = measurements.iloc[step]['compass']
            rotation_matrix = np.array([
                [np.cos(compass), -np.sin(compass)],
                [np.sin(compass), np.cos(compass)]
            ])
            target = measurements.iloc[step]['target'].asarray()
            gps = measurements.iloc[step]['gps']
            target = rotation_matrix.T.dot(target - gps)
            metrics = np.concatenate(([speed], target))
            metrics_ep.append(metrics)
            reward = 0
            rewards_ep.append(reward)
            states_ep.append(rgb)
            pass
        states.append(states_ep)
        actions.append(actions_ep)
        rewards.append(rewards_ep)
        metrics.append(metrics_ep)
        lens.append(len(states_ep))

    states = torch.as_tensor(states).float()
    metrics = torch.as_tensor(metrics).float()
    actions = torch.as_tensor(actions).float()
    lens = torch.as_tensor(lens).long()
    rewards = torch.as_tensor(rewards).long()
    data = {
        'states': states,
        'actions': actions,
        'metrics': metrics,
        'lengths': lens,
        'rewards': rewards
    }
    if file_path:
        torch.save(data, file_path)


if __name__ == "__main__":
    save_dataset('gail_experts/trajs_carla.pt')