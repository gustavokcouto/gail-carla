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
    def __init__(self, obs_shape, metrics_space, action_space):
        super(Policy, self).__init__()

        self.base = CNNBase(obs_shape[0])

        num_outputs = action_space.shape[0]
        self.dist = DiagGaussian(self.base.output_size, num_outputs)

    def act(self, obs, metrics, masks, deterministic=False):
        value, actor_features = self.base(obs, metrics, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, obs, metrics, masks):
        value, _ = self.base(obs, metrics, masks)
        return value

    def evaluate_actions(self, obs, metrics, masks, action):
        value, actor_features = self.base(obs, metrics, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=512):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.image_encoder = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=2)), nn.ReLU(), Flatten())
        
        self.state_encoder = nn.Sequential(
            init_(nn.Linear(64 * 7 * 7 + 2, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self._hidden_size = hidden_size

        self.train()

    def forward(self, obs, metrics, masks):
        x = self.image_encoder(obs)
        x = self.state_encoder(torch.cat([x, metrics], dim=1))

        return self.critic_linear(x), x

    @property
    def output_size(self):
        return self._hidden_size