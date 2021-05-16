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

        num_outputs = action_space.shape[0]
        self.base = CNNBase(obs_shape, metrics_space, num_outputs)

        std_dev = np.array([[0.06, 0.05]], dtype=np.float32)
        self.std_dev = torch.Tensor(std_dev)
        self.max = torch.Tensor([1, 1])
        self.min = torch.Tensor([-1, 0])
    def act(self, obs, metrics, deterministic=False):
        value, output = self.base(obs, metrics)
        std_dev = self.std_dev.to(output)
        dist = torch.distributions.Normal(output, std_dev)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        
        action = torch.max(torch.min(action, self.max.to(output)), self.min.to(output))
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        return value, action, action_log_probs

    def get_value(self, obs, metrics):
        value, _ = self.base(obs, metrics)
        return value

    def evaluate_actions(self, obs, metrics, action):
        value, output = self.base(obs, metrics)
        std_dev = self.std_dev.to(output)
        dist = torch.distributions.Normal(output, std_dev)
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action_log_probs, dist_entropy


class CNNBase(nn.Module):
    def __init__(self, obs_shape, metrics_space, num_outputs, hidden_size=512):
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
        self.output_linear = init_(nn.Linear(hidden_size, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

        self.train()

    def forward(self, obs, metrics):
        x = self.main(obs)
        x = self.trunk(torch.cat([x, metrics], dim=1))
        critic = self.critic_linear(x)
        output = self.output_linear(x)

        return critic, output
