import sys
import json

from pathlib import Path

import numpy as np
import tqdm
import carla
import pandas as pd

from PIL import Image

from carla_env import CarlaEnv
from auto_pilot.auto_pilot import AutoPilot

from auto_pilot.route_parser import parse_routes_file
from auto_pilot.route_manipulation import interpolate_trajectory
from tools.envs import EnvMonitor

def gen_trajectories(routes_file=''):
    # Instantiate the env
    np.random.seed(1337)
    ep_max_len = 2400
    host = 'localhost'
    port = 2000
    env = CarlaEnv(host, port, ep_max_len, routes_file, train=False, eval=True, env_id='expert')
    
    expert_file_dir = Path('gail_experts') / routes_file.stem
    expert_file_dir.mkdir(parents=True)

    env = EnvMonitor(env, output_path=expert_file_dir)

    for route_id in [0]:
        env.env.set_route(route_id)
        trajectory = env.env.trajectory
        global_plan_gps, global_plan_world_coord = interpolate_trajectory(env.env._world, trajectory)
        for ep_id in tqdm.tqdm(range(30)):
            episode_dir = expert_file_dir / ('route_%02d' % route_id) / ('ep_%02d' % ep_id)
            (episode_dir / 'rgb').mkdir(parents=True)
            (episode_dir / 'rgb_left').mkdir(parents=True)
            (episode_dir / 'rgb_right').mkdir(parents=True)
            (episode_dir / 'topdown').mkdir(parents=True)
            metrics_ep = []
            actions_ep = []

            i_step = 0
            _, step_metrics = env.reset()
            auto_pilot = AutoPilot(global_plan_gps, global_plan_world_coord)
            while not env.env.route_completed and i_step < ep_max_len:
                ego_metrics = [
                    env.env.info['gps_x'],
                    env.env.info['gps_y'],
                    env.env.info['compass'],
                    env.env.info['speed']
                ]
                action = auto_pilot.run_step(ego_metrics)

                metrics_ep.append(step_metrics.numpy())
                actions_ep.append(action)
                Image.fromarray(env.env.rgb_left).save(episode_dir / 'rgb_left' / ('%04d.png' % i_step))
                Image.fromarray(env.env.rgb).save(episode_dir / 'rgb' / ('%04d.png' % i_step))
                Image.fromarray(env.env.rgb_right).save(episode_dir / 'rgb_right' / ('%04d.png' % i_step))
                Image.fromarray(env.env.topdown).save(episode_dir / 'topdown' / ('%04d.png' % i_step))

                _, step_metrics, _, _, _ = env.step(action)
                i_step += 1

            metrics_ep.append(step_metrics.numpy())
            actions_ep.append(action)

            Image.fromarray(env.env.rgb_left).save(episode_dir / 'rgb_left' / ('%04d.png' % i_step))
            Image.fromarray(env.env.rgb).save(episode_dir / 'rgb' / ('%04d.png' % i_step))
            Image.fromarray(env.env.rgb_right).save(episode_dir / 'rgb_right' / ('%04d.png' % i_step))
            Image.fromarray(env.env.topdown).save(episode_dir / 'topdown' / ('%04d.png' % i_step))

            ep_df = pd.DataFrame({
                'actions': actions_ep,
                'metrics': metrics_ep,
            })
            ep_df.to_json(episode_dir / 'episode.json')

if __name__ == "__main__":
    gen_trajectories(Path('data/routes_training.xml'))
    pass