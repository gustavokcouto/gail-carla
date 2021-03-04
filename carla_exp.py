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


def gen_trajectories(file_path=''):
    # Instantiate the env
    np.random.seed(1337)
    env = CarlaEnv()

    route_file = Path('data/route_00.xml')
    trajectory = parse_routes_file(route_file)
    global_plan_gps, global_plan_world_coord = interpolate_trajectory(env._world, trajectory)

    # Test the trained agent
    n_episodes = 10
    n_steps = 800
    states = []
    actions = []
    rewards = []

    lens = []
    for _ in range(n_episodes):
        states_ep = []
        actions_ep = []
        rewards_ep = []
        obs = env.reset()
        auto_pilot = AutoPilot(global_plan_gps, global_plan_world_coord)
        states_ep.append(obs)
        for step in range(n_steps):
            ego_metrics = [
                env.info['gps_x'],
                env.info['gps_y'],
                env.info['compass'],
                env.info['speed']
            ]

            action = auto_pilot.run_step(ego_metrics)
            actions_ep.append(action)
            obs, reward, _, _ = env.step(action)
            reward = 0
            rewards_ep.append(reward)
            states_ep.append(obs)
        states.append(states_ep)
        actions.append(actions_ep)
        rewards.append(rewards_ep)
        lens.append(step + 1)

    states = torch.as_tensor(states).float()
    actions = torch.as_tensor(actions).float()
    lens = torch.as_tensor(lens).long()
    rewards = torch.as_tensor(rewards).long()
    data = {
        'states': states,
        'actions': actions,
        'lengths': lens,
        'rewards': rewards
    }
    if file_path:
        torch.save(data, file_path)

if __name__ == "__main__":
    gen_trajectories('gail_experts/trajs_carla.pt')
    pass