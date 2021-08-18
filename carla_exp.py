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
from tools.envs import EnvMonitor

def gen_trajectories(route_file='', save_obs=True):
    # Instantiate the env
    np.random.seed(1337)
    ep_len = 2400
    host = 'localhost'
    port = 2000
    env = CarlaEnv(host, port, ep_len, route_file, train=False, eval=True, env_id='expert')
    
    expert_file_dir = Path('gail_experts') / route_file.stem
    expert_file_dir.mkdir(parents=True)

    env = EnvMonitor(env, output_path=expert_file_dir)

    trajectory = parse_routes_file(route_file)
    global_plan_gps, global_plan_world_coord = interpolate_trajectory(env.env._world, trajectory)

    # Test the trained agent
    n_episodes = 12
    states = []
    metrics = []
    actions = []
    rewards = []

    lens = []
    max_len = 0
    for episode in tqdm.tqdm(range(n_episodes)):
        episode_dir = expert_file_dir / ('episode_%02d' % episode)
        (episode_dir / 'rgb').mkdir(parents=True)
        (episode_dir / 'rgb_left').mkdir(parents=True)
        (episode_dir / 'rgb_right').mkdir(parents=True)
        (episode_dir / 'topdown').mkdir(parents=True)
        states_ep = []
        metrics_ep = []
        actions_ep = []
        rewards_ep = []

        i_step = 0
        obs, step_metrics = env.reset()
        auto_pilot = AutoPilot(global_plan_gps, global_plan_world_coord)
        while not env.env.route_completed:
            ego_metrics = [
                env.env.info['gps_x'],
                env.env.info['gps_y'],
                env.env.info['compass'],
                env.env.info['speed']
            ]
            action = auto_pilot.run_step(ego_metrics)

            metrics_ep.append(step_metrics)
            reward = 0
            rewards_ep.append(reward)
            if save_obs:
                states_ep.append(obs)
            actions_ep.append(action)
            Image.fromarray(env.env.rgb_left).save(episode_dir / 'rgb_left' / ('%04d.png' % i_step))
            Image.fromarray(env.env.rgb).save(episode_dir / 'rgb' / ('%04d.png' % i_step))
            Image.fromarray(env.env.rgb_right).save(episode_dir / 'rgb_right' / ('%04d.png' % i_step))
            Image.fromarray(env.env.topdown).save(episode_dir / 'topdown' / ('%04d.png' % i_step))

            obs, step_metrics, reward, _, _ = env.step(action)
            i_step += 1

        metrics_ep.append(step_metrics)
        reward = 0
        rewards_ep.append(reward)
        if save_obs:
            states_ep.append(obs)

        Image.fromarray(env.env.rgb_left).save(episode_dir / 'rgb_left' / ('%04d.png' % i_step))
        Image.fromarray(env.env.rgb).save(episode_dir / 'rgb' / ('%04d.png' % i_step))
        Image.fromarray(env.env.rgb_right).save(episode_dir / 'rgb_right' / ('%04d.png' % i_step))
        Image.fromarray(env.env.topdown).save(episode_dir / 'topdown' / ('%04d.png' % i_step))

        ep_len = len(actions_ep)
        if ep_len > max_len:
            max_len = ep_len
        else:
            for _ in range(max_len - ep_len):
                metrics_ep.append(step_metrics)
                rewards_ep.append(reward)
                if save_obs:
                    states_ep.append(obs)
                actions_ep.append(action)

        if save_obs:
            states.append(states_ep)
        actions.append(actions_ep)
        rewards.append(rewards_ep)
        metrics.append(metrics_ep)
        lens.append(ep_len)

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

    if save_obs:
        states = torch.as_tensor(states).float()
        data['states'] = states

    expert_file = expert_file_dir / 'trajs_carla.pt'
    torch.save(data, expert_file)

if __name__ == "__main__":
    gen_trajectories(Path('data/route_00.xml'))
    pass