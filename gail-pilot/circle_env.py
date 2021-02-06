import math
import numpy as np
import gym
from gym import spaces


class CircleEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left. 
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(CircleEnv, self).__init__()

        self.action_space = spaces.Box(low=-10, high=10,
                                            shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(10,), dtype=np.float32)

    def reset(self):
        self.center_x = 0.0
        self.center_y = 0.3
        self.episode_reward = 0
        self.speed = 0.01
        self.angle = 0.0
        self.cur_length = 0
        self.ep_length = 1000
        self.hist5 = []
        for _ in range(5):
            pos_x = np.random.normal(scale=0.001)
            pos_y = np.random.normal(scale=0.001)
            self.hist5.append(pos_x)
            self.hist5.append(pos_y)

        return np.array(self.hist5).astype(np.float64)

    def step(self, action):
        pos_x = self.hist5[-2]
        pos_y = self.hist5[-1]
        action_angle = math.atan2(action[1], action[0])
        pos_x += self.speed * math.cos(action_angle)
        pos_y += self.speed * math.sin(action_angle)
        pos_x += np.random.normal(scale=0.001)
        pos_y += np.random.normal(scale=0.001)

        for i in range(0, len(self.hist5) - 2):
            self.hist5[i] = self.hist5[i + 2]
            self.hist5[-2] = pos_x
            self.hist5[-1] = pos_y
        done = False
        self.cur_length += 1
        reward = - abs(
            (self.center_x ** 2 + self.center_y ** 2) ** 0.5
            - ((pos_x - self.center_x) ** 2 + (pos_y - self.center_y) ** 2) ** 0.5
        ) / (self.center_x ** 2 + self.center_y ** 2) ** 0.5
        self.episode_reward += reward
        info = {}
        if self.cur_length >= self.ep_length:
            info['episode'] = {'r': self.episode_reward}
            done = True

        return np.array(self.hist5.copy()).astype(np.float64), reward, done, info

    def close(self):
        pass