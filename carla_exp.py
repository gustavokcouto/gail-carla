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
from carla_gym.core.task_actor.scenario_actor.agents.basic_agent import BasicAgent
from carla_gym.utils.expert_noiser import ExpertNoiser


def gen_trajectories(routes_file=''):
    # Instantiate the env
    np.random.seed(1337)
    ep_max_len = 6000
    host = '192.168.0.4'
    port = 2000
    env = CarlaEnv(host, port, ep_max_len, routes_file, train=False, eval=True, env_id='expert')
    
    expert_file_dir = Path('gail_experts') / routes_file.stem
    expert_file_dir.mkdir(parents=True)
    longitudinal_noiser = ExpertNoiser('Throttle', frequency=15, intensity=10, min_noise_time_amount=2.0)
    lateral_noiser = ExpertNoiser('Spike', frequency=25, intensity=4, min_noise_time_amount=0.5)
    for route_id in tqdm.tqdm(range(10)):
        env.env.set_task_idx(route_id)
        for ep_id in range(1):
            episode_dir = expert_file_dir / ('route_%02d' % route_id) / ('ep_%02d' % ep_id)
            (episode_dir / 'rgb').mkdir(parents=True)
            (episode_dir / 'rgb_left').mkdir(parents=True)
            (episode_dir / 'rgb_right').mkdir(parents=True)
            (episode_dir / 'birdview').mkdir(parents=True)
            (episode_dir / 'birdview_masks').mkdir(parents=True)
            metrics_ep = []
            actions_ep = []

            i_step = 0
            _, step_metrics = env.reset()
            basic_agent = BasicAgent(env.env._ev_handler.ego_vehicles['hero'], None, 6.0)
            while not env.route_completed:
                action = basic_agent.get_action()
                action = longitudinal_noiser.compute_noise(action, step_metrics.numpy()[2] * 3.6)
                action = lateral_noiser.compute_noise(action, step_metrics.numpy()[2] * 3.6)

                metrics_ep.append(step_metrics.numpy())
                actions_ep.append(action)
                Image.fromarray(env.rgb_left).save(episode_dir / 'rgb_left' / '{:0>4d}.png'.format(i_step))
                Image.fromarray(env.rgb).save(episode_dir / 'rgb' / '{:0>4d}.png'.format(i_step))
                Image.fromarray(env.rgb_right).save(episode_dir / 'rgb_right' / '{:0>4d}.png'.format(i_step))
                Image.fromarray(env.birdview).save(episode_dir / 'birdview' / '{:0>4d}.png'.format(i_step))
                for i_mask, birdview_mask in enumerate(env.birdview_masks):
                    Image.fromarray(birdview_mask).save(episode_dir / 'birdview_masks' / '{:0>4d}_{:0>2d}.png'.format(i_step, i_mask))
                _, step_metrics, _, _, _ = env.step(action)
                i_step += 1

            metrics_ep.append(step_metrics.numpy())
            actions_ep.append(action)

            Image.fromarray(env.rgb_left).save(episode_dir / 'rgb_left' / '{:0>4d}.png'.format(i_step))
            Image.fromarray(env.rgb).save(episode_dir / 'rgb' / '{:0>4d}.png'.format(i_step))
            Image.fromarray(env.rgb_right).save(episode_dir / 'rgb_right' / '{:0>4d}.png'.format(i_step))
            Image.fromarray(env.birdview).save(episode_dir / 'birdview' / '{:0>4d}.png'.format(i_step))
            for i_mask, birdview_mask in enumerate(env.birdview_masks):
                    Image.fromarray(birdview_mask).save(episode_dir / 'birdview_masks' / '{:0>4d}_{:0>2d}.png'.format(i_step, i_mask))
            ep_df = pd.DataFrame({
                'actions': actions_ep,
                'metrics': metrics_ep,
            })

            ep_df.to_json(episode_dir / 'episode.json')

if __name__ == "__main__":
    gen_trajectories(Path('data/routes_training.xml'))
    pass