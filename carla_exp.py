import sys
import json

from pathlib import Path

import numpy as np
import tqdm
import carla
import cv2
import torch
import pandas as pd

from PIL import Image

from carla_env import CarlaEnv
from auto_pilot.auto_pilot import AutoPilot

from auto_pilot.route_parser import parse_routes_file
from auto_pilot.route_manipulation import interpolate_trajectory

FRAME_SKIP = 1
EPISODE_LENGTH = 800

def gen_trajectories(file_path=''):
    # Instantiate the env
    np.random.seed(1337)
    env = CarlaEnv()

    route_file = Path('data/route_00.xml')
    trajectory = parse_routes_file(route_file)
    global_plan_gps, global_plan_world_coord = interpolate_trajectory(env._world, trajectory)

    # Test the trained agent
    n_episodes = 4
    states = []
    metrics = []
    actions = []
    rewards = []

    lens = []
    for episode in range(n_episodes):
        states_ep = []
        metrics_ep = []
        actions_ep = []
        rewards_ep = []

        obs, step_metrics = env.reset()
        auto_pilot = AutoPilot(global_plan_gps, global_plan_world_coord)
        for step in tqdm.tqdm(range(EPISODE_LENGTH * FRAME_SKIP)):
            ego_metrics = [
                env.info['gps_x'],
                env.info['gps_y'],
                env.info['compass'],
                env.info['speed']
            ]
            action = auto_pilot.run_step(ego_metrics)
            if step % FRAME_SKIP == 0:
                metrics_ep.append(step_metrics)
                reward = 0
                rewards_ep.append(reward)
                states_ep.append(obs)
                actions_ep.append(action)

            obs, step_metrics, reward, _, _ = env.step(action)
        metrics_ep.append(step_metrics)
        reward = 0
        rewards_ep.append(reward)
        states_ep.append(obs)

        states.append(states_ep)
        actions.append(actions_ep)
        rewards.append(rewards_ep)
        metrics.append(metrics_ep)
        lens.append(len(actions_ep))

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
    gen_trajectories('gail_experts/trajs_carla.pt')
    pass