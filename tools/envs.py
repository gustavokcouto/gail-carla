from pathlib import Path
import numpy as np
import torch
import pandas as pd

from carla_env import CarlaEnv
from vec_env.vec_env import VecEnvWrapper
from vec_env.subproc_vec_env import SubprocVecEnv
from common.running_mean_std import RunningMeanStd


class EnvEpoch():
    envs_epoch = 0

    @classmethod
    def set_epoch(cls, i_epoch):
        cls.envs_epoch = i_epoch

    @classmethod
    def get_epoch(cls):
        return cls.envs_epoch


def make_env(env_host, env_port, ep_length, route_file, env_id, route_id):
    def _thunk():
        env = CarlaEnv(env_host, env_port, ep_length, route_file, env_id=env_id, train=True, route_id=route_id)

        env = EnvMonitor(env)

        return env

    return _thunk


def make_vec_envs(envs_params, device, ep_length, route_file, routes):
    envs = [
        make_env(env_params['host'], env_params['port'], ep_length, route_file, 'train_env_{}'.format(env_id), routes[env_id % len(routes)])
        for env_id, env_params in enumerate(envs_params)
    ]

    envs = SubprocVecEnv(envs)

    envs = VecPyTorch(envs, device)

    return envs


class EnvMonitor():
    def __init__(self, env, output_path=None) -> None:
        self.env = env
        self.env_id = env.env_id
        self.ep_df = pd.DataFrame()
        self.ep_count = 0
        if not output_path:
            self.output_path = Path("runs/env_info")
        else:
            self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.metrics_space = self.env.metrics_space
        self.spec = self.env.spec
        self.ep_length = self.env.ep_length

    def step(self, action):
        obs, metrics, reward, done, info = self.env.step(action)
        info['ep_count'] = self.ep_count
        info['i_epoch'] = EnvEpoch.get_epoch()
        self.ep_df = self.ep_df.append(info, ignore_index=True)
        return obs, metrics, reward, done, info
    
    def reset(self):
        obs, metrics = self.env.reset()
        if self.ep_count < 2:
            self.ep_df.to_csv(
                self.output_path / '{}.csv'.format(self.env_id),
                index=False
            )
        else:
            self.ep_df.to_csv(
                self.output_path / '{}.csv'.format(self.env_id),
                index=False,
                mode='a',
                header=False
            )
        self.ep_count += 1
        self.ep_df = pd.DataFrame()
        return obs, metrics

    def close(self):
        self.env.close()
        pass


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs, metrics = self.venv.reset()
        obs = obs.to(self.device)
        metrics = metrics.to(self.device)
        return obs, metrics

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, metrics, reward, done, info = self.venv.step_wait()
        obs = obs.to(self.device)
        metrics = metrics.to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        return obs, metrics, reward, done, info