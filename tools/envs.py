import torch

from carla_env import CarlaEnv
from vec_env.vec_env import VecEnvWrapper
from vec_env.subproc_vec_env import SubprocVecEnv


def make_env(env_id):
    def _thunk():
        env = CarlaEnv(env_id)

        return env

    return _thunk


def make_vec_envs(num_processes,
                  device):
    envs = [
        make_env(i)
        for i in range(num_processes)
    ]

    envs = SubprocVecEnv(envs)

    envs = VecPyTorch(envs)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv):
        super(VecPyTorch, self).__init__(venv)

    def reset(self):
        obs, metrics = self.venv.reset()
        obs = torch.from_numpy(obs).float()
        metrics = torch.from_numpy(metrics).float()
        return obs, metrics

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, metrics, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float()
        metrics = torch.from_numpy(metrics).float()
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, metrics, reward, done, info