import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.distributions import DiagGaussian
from tools.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, metrics_space, action_space, activation=None):
        super(Policy, self).__init__()

        self.base = CNNBase(obs_shape, metrics_space)

        num_outputs = action_space.shape[0]
        self.dist = DiagGaussian(self.base.output_size, num_outputs, activation=activation)

    def act(self, obs, metrics, deterministic=False):
        value, actor_features = self.base(obs, metrics)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, obs, metrics):
        value, _ = self.base(obs, metrics)
        return value

    def evaluate_actions(self, obs, metrics, action):
        value, actor_features = self.base(obs, metrics)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class CNNBase(nn.Module):
    def __init__(self, obs_shape, metrics_space, hidden_size=512):
        super(CNNBase, self).__init__()

        C, H, W = obs_shape

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(C, 32, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 128, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(128, 256, 4, stride=2)), nn.ReLU(), Flatten())

        for i in range(4):
            H = (H - 4)//2 + 1
            W = (W - 4)//2 + 1
        # Get image dim
        img_dim = 256*H*W


        self.trunk = nn.Sequential(
            init_(nn.Linear(metrics_space.shape[0] + img_dim, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self._hidden_size = hidden_size

        self.train()

    def forward(self, obs, metrics):
        x = self.main(obs)
        x = self.trunk(torch.cat([x, metrics], dim=1))

        return self.critic_linear(x), x

    @property
    def output_size(self):
        return self._hidden_size
