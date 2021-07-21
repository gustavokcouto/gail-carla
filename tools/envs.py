import numpy as np
import torch

from carla_env import CarlaEnv
from vec_env.vec_env import VecEnvWrapper
from vec_env.subproc_vec_env import SubprocVecEnv
from common.running_mean_std import RunningMeanStd


def make_env(ep_length, route_file, env_id):
    def _thunk():
        env = CarlaEnv(ep_length, route_file, env_id,)

        return env

    return _thunk


def make_vec_envs(num_processes, device, ep_length, route_file):
    envs = [
        make_env(ep_length, route_file, i)
        for i in range(num_processes)
    ]

    envs = SubprocVecEnv(envs)

    envs = VecNormalize(envs)

    envs = VecPyTorch(envs, device)

    return envs


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, metrics, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, metrics, rews, news, infos

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs, metrics = self.venv.reset()
        return obs, metrics


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs, metrics = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        metrics = torch.from_numpy(metrics).float().to(self.device)
        return obs, metrics

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, metrics, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        metrics = torch.from_numpy(metrics).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        return obs, metrics, reward, done, info