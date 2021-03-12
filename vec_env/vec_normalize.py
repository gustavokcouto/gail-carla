from vec_env.vec_env import VecEnvWrapper
from common.running_mean_std import RunningMeanStd
import numpy as np


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.metrics_rms = RunningMeanStd(shape=self.metrics_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, metrics, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        metrics = self._metricsfilt(metrics)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, metrics, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def _metricsfilt(self, metrics):
        if self.metrics_rms:
            self.metrics_rms.update(metrics)
            metrics = np.clip((metrics - self.metrics_rms.mean) / np.sqrt(self.metrics_rms.var + self.epsilon), -self.clipob, self.clipob)
            return metrics
        else:
            return metrics

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs, metrics = self.venv.reset()
        obs = self._obfilt(obs)
        metrics = self._metricsfilt(metrics)
        return obs, metrics