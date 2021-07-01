import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, metrics_space, action_space, activation, std_dev, var_ent):
        super(Policy, self).__init__()

        num_outputs = action_space.shape[0]
        self.base = CNNBase(obs_shape, metrics_space, num_outputs, activation, std_dev, var_ent)

        self.max = torch.Tensor([1, 1])
        self.min = torch.Tensor([-1, 0])

    def act(self, obs, metrics, deterministic=False):
        value, output, logstd = self.base(obs, metrics)
        dist = torch.distributions.Normal(output, logstd.exp())
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        # action = torch.max(torch.min(action, self.max.to(action)), self.min.to(action))
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        return value, action, action_log_probs

    def set_epoch(self,epoch):
        self.base.set_epoch(epoch)

    def get_value(self, obs, metrics):
        value, _, _ = self.base(obs, metrics)
        return value

    def evaluate_actions(self, obs, metrics, action):
        value, output, logstd = self.base(obs, metrics)
        dist = torch.distributions.Normal(output, logstd.exp())

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()
        steer_std = logstd[0, 0].detach().cpu()
        throttle_std = logstd[0, 1].detach().cpu()
        return value, action_log_probs, dist_entropy, steer_std, throttle_std


class CNNBase(nn.Module):
    def __init__(self, obs_shape, metrics_space, num_outputs, activation, std_dev, var_ent, hidden_size=512):
        super(CNNBase, self).__init__()

        C, H, W = obs_shape

        self.main = nn.Sequential(
            nn.Conv2d(C, 32, 4, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.LeakyReLU(0.2),
            Flatten()
        )

        for i in range(4):
            H = (H - 4)//2 + 1
            W = (W - 4)//2 + 1
        # Get image dim
        img_dim = 256*H*W


        self.trunk = nn.Sequential(
            nn.Linear(metrics_space.shape[0] + img_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1 + num_outputs)
        )

        self.std_dev = std_dev
        self.var_ent = var_ent
        if var_ent:
            self.logstd = nn.Parameter(torch.Tensor([std_dev[0]['logstd']]))

        self.activation = activation
        self.train()

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.var_ent:
            for std_dev in self.std_dev:
                if 'limit' in std_dev and self.epoch > std_dev['limit']:
                    continue
                self.logstd = torch.Tensor(std_dev['logstd'])
                break

    def forward(self, obs, metrics):
        x = self.main(obs)
        nn_output = self.trunk(torch.cat([x, metrics], dim=1))
        critic = nn_output[...,0].unsqueeze(dim=1)
        output = nn_output[...,1:]
        if self.activation:
            output[...,0] = torch.tanh(output[...,0])
            output[...,1] = torch.sigmoid(output[...,1])
        zeros = torch.zeros(output.size()).to(output)
        logstd = self.logstd.to(output)
        logstd = logstd + zeros
        return critic, output, logstd

