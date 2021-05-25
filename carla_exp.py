import sys
import json

from pathlib import Path

import numpy as np
import tqdm
import carla
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
    env = CarlaEnv(train=False)

    route_file = Path('data/route_00.xml')
    trajectory = parse_routes_file(route_file)
    global_plan_gps, global_plan_world_coord = interpolate_trajectory(env._world, trajectory)

    # Test the trained agent
    n_episodes = 15
    states = []
    metrics = []
    actions = []
    rewards = []

    lens = []
    max_len = 0
    for episode in tqdm.tqdm(range(n_episodes)):
        states_ep = []
        metrics_ep = []
        actions_ep = []
        rewards_ep = []

        obs, step_metrics = env.reset()
        auto_pilot = AutoPilot(global_plan_gps, global_plan_world_coord)
        while not env.route_completed:
            ego_metrics = [
                env.info['gps_x'],
                env.info['gps_y'],
                env.info['compass'],
                env.info['speed']
            ]
            action = auto_pilot.run_step(ego_metrics)

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
        ep_len = len(actions_ep)
        if ep_len > max_len:
            max_len = ep_len
        else:
            for _ in range(max_len - ep_len):
                metrics_ep.append(step_metrics)
                rewards_ep.append(reward)
                states_ep.append(obs)
                actions_ep.append(action)

        states.append(states_ep)
        actions.append(actions_ep)
        rewards.append(rewards_ep)
        metrics.append(metrics_ep)
        lens.append(ep_len)
        


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