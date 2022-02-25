import math
import numpy as np
import gym
from gym import spaces
import collections
import queue
import time
import torch
from torchvision import transforms

import carla
from carla_gym.envs import LeaderboardEnv

from PIL import Image, ImageDraw


obs_configs = {
    'hero': {
        'speed': {
            'module': 'actor_state.speed'
        },
        'gnss': {
            'module': 'navigation.gnss'
        },
        'central_rgb': {
            'module': 'camera.rgb',
            'fov': 60,
            'width': 384,
            'height': 216,
            'location': [0.8, 0.0, 1.3],
            'rotation': [0.0, 0.0, 0.0]
        },
        'left_rgb': {
            'module': 'camera.rgb',
            'fov': 60,
            'width': 384,
            'height': 216,
            'location': [0.8, 0.0, 1.3],
            'rotation': [0.0, 0.0, -55.0]
        },
        'right_rgb': {
            'module': 'camera.rgb',
            'fov': 60,
            'width': 384,
            'height': 216,
            'location': [0.8, 0.0, 1.3],
            'rotation': [0.0, 0.0, 55.0]
        },
        'birdview': {
            'module': 'birdview.chauffeurnet',
            'width_in_pixels': 192,
            'pixels_ev_to_bottom': 40,
            'pixels_per_meter': 5.0,
            'history_idx': [-16, -11, -6, -1],
            'scale_bbox': False,
        },
        'route_plan': {
            'module': 'navigation.waypoint_plan',
            'steps': 20
        }
    }
}
reward_configs = {
    'hero': {
        'entry_point': 'reward.valeo_action:ValeoAction'
    }
}
terminal_configs = {
    'hero': {
        'entry_point': 'terminal.leaderboard:Leaderboard',
    }
}
env_configs = {
    'carla_map': 'Town01',
    'weather_group': 'train',
    'routes_group': 'train'
}



class CarlaEnv(gym.Env):
    def __init__(self, host, port, ep_length, routes_file, train=False, eval=False, env_id=None, route_id=0):
        super(CarlaEnv, self).__init__()

        self.env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                   terminal_configs=terminal_configs, host=host, port=port,
                   seed=2021, no_rendering=False, train=train, **env_configs)
        self.env_id = env_id
        self.ep_length = ep_length
        self.env.set_task_idx(route_id)
        self.route_id = route_id

        self.action_space = spaces.Box(low=-10, high=10,
                                       shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(9, 216, 384), dtype=np.uint8)

        self.metrics_space = spaces.Box(low=-100, high=100,
                                        shape=(4,), dtype=np.float32)

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def reset(self):
        self.env.reset()
        self.cur_length = 0
        self.route_completed = False
        self.last_total_reward = 0
        self.episode_reward = 0

        obs, metrics, _, _, _ = self.step(None)

        self.episode_reward = 0

        return obs, metrics

    def step(self, action):
        control = carla.VehicleControl()

        if action is not None:
            control.steer = float(action[0])
            control.throttle = float(action[1])

        driver_control = {'hero': control}
        new_obs, reward, done, info = self.env.step(driver_control)

        self.rgb = new_obs['hero']['central_rgb']['data']
        self.rgb_left = new_obs['hero']['left_rgb']['data']
        self.rgb_right = new_obs['hero']['right_rgb']['data']
        self.birdview = new_obs['hero']['birdview']['rendered']
        self.birdview_masks = []
        for i_channels in range(5):
            birdview_mask = new_obs['hero']['birdview']['masks'][i_channels * 3: i_channels * 3 + 3, :, :]
            self.birdview_masks.append(np.transpose(birdview_mask, [1, 2, 0]).astype(np.uint8))
        rgb = Image.fromarray(self.rgb).convert("RGB")
        rgb_left = Image.fromarray(self.rgb_left).convert("RGB")
        rgb_right = Image.fromarray(self.rgb_right).convert("RGB")
        rgb = self.preprocess(rgb)
        rgb_left = self.preprocess(rgb_left)
        rgb_right = self.preprocess(rgb_right)
        obs = torch.cat([rgb, rgb_left, rgb_right])

        speed = new_obs['hero']['speed']['speed']
        target_gps = new_obs['hero']['gnss']['target_gps']
        command = new_obs['hero']['gnss']['command']

        metrics = torch.Tensor([target_gps[0], target_gps[1], speed[0], command[0]])

        self.cur_length += 1

        route_completion = info['hero']['route_completion']['route_completed_in_m']
        route_length = info['hero']['route_completion']['route_length_in_m']
        total_reward = route_completion / route_length
        reward = total_reward - self.last_total_reward
        self.last_total_reward = total_reward
        self.episode_reward += reward

        self.route_completed = info['hero']['route_completion']['is_route_completed']
        if info['hero']['route_deviation'] is not None:
            print("Route deviation detected.\n")
        if info['hero']['collision'] is not None:
            print("Collision detected.\n")

        info = {}
        info['route_id'] = self.route_id
        info['episode_reward'] = self.episode_reward
        done = done['hero']
        if done:
            info['episode'] = {'r': self.episode_reward, 'l': self.cur_length}

        self.info = info
        reward = np.array(reward)

        return obs, metrics, reward, done, info

    def close(self):
        pass
