import numpy as np
import torch
import torch.nn as nn

from tools.resnet import resnet18


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

        self.obs_processor = ProcessObsFeaturesResnet(obs_shape, bias=True)
        self.metrics_processor = ProcessMetrics(metrics_space.shape[0])
        
        self.trunk = nn.Sequential(
            nn.Linear(self.obs_processor.output_dim + self.metrics_processor.output_dim, hidden_size),
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
        obs_features, _ = self.obs_processor(obs)
        metrics_features, _ = self.metrics_processor(metrics)

        nn_output = self.trunk(torch.cat([obs_features, metrics_features], dim=1))
        critic = nn_output[...,0].unsqueeze(dim=1)
        output = nn_output[...,1:]
        if self.activation:
            output[...,0] = torch.tanh(output[...,0])
            output[...,1] = torch.sigmoid(output[...,1])
        zeros = torch.zeros(output.size()).to(output)
        logstd = self.logstd.to(output)
        logstd = logstd + zeros
        return critic, output, logstd



class ProcessObsFeatures(nn.Module):
    def __init__(self, obs_shape, bias=True):
        super(ProcessObsFeatures, self).__init__()
        C, H, W = obs_shape

        self.main = nn.Sequential(
            nn.Conv2d(C, 32, 4, stride=2, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, bias=bias),
            nn.LeakyReLU(0.2),
            Flatten()
        )

        for _ in range(4):
            H = (H - 4)//2 + 1
            W = (W - 4)//2 + 1

        # Get image dim
        self.output_dim = 256*H*W

    def forward(self, obs):
        # scale observation
        obs_transformed = obs / 255
        obs_transformed.requires_grad = True
        obs_features = self.main(obs_transformed)

        return obs_features, obs_transformed


class ProcessObsFeaturesResnet(nn.Module):
    def __init__(self, obs_shape, bias=True):
        super(ProcessObsFeaturesResnet, self).__init__()
        input_channels, _, _ = obs_shape
        self.output_dim = 512

        self.main = resnet18(input_channels)

    def forward(self, obs):
        # scale observation
        obs_transformed = obs / 255
        obs_transformed.requires_grad = True
        obs_features = self.main(obs_transformed)

        return obs_features, obs_transformed


class ProcessMetrics(nn.Module):
    def __init__(self, metrics_shape):
        super(ProcessMetrics, self).__init__()

        target_embedding_dimension = 8
        self.target_disc_space = 1000
        self.target_x_embedding = nn.Embedding(self.target_disc_space, target_embedding_dimension)
        self.target_y_embedding = nn.Embedding(self.target_disc_space, target_embedding_dimension)

        speed_embedding_dimension = 8
        self.speed_disc_space = 1000
        self.speed_embedding = nn.Embedding(self.speed_disc_space, speed_embedding_dimension)

        road_option_embedding_dimension = 8
        max_road_options = 10
        self.road_option_embedding = nn.Embedding(max_road_options, road_option_embedding_dimension)

        self.output_dim = 2 * target_embedding_dimension + speed_embedding_dimension + road_option_embedding_dimension

    def forward(self, metrics):
        # metrics composition [target[0], target[1], speed, int(road_option)]

        # max target of 0.001
        target_buckets = np.linspace(-0.001, 0.001, num=self.target_disc_space)
        target_x_disc = np.digitize(metrics[:, 0].cpu(), target_buckets)
        target_y_disc = np.digitize(metrics[:, 1].cpu(), target_buckets)
        target_x_disc = torch.from_numpy(target_x_disc).long().to(metrics.device)
        target_y_disc = torch.from_numpy(target_y_disc).long().to(metrics.device)
        target_x_features = self.target_x_embedding(target_x_disc)
        target_y_features = self.target_y_embedding(target_y_disc)

        # max of 60m/s or 216km/h
        speed_buckets = np.linspace(-60, 60, num=self.speed_disc_space)
        speed_disc = np.digitize(metrics[:, 2].cpu(), speed_buckets)
        speed_disc = torch.from_numpy(speed_disc).long().to(metrics.device)
        speed_features = self.speed_embedding(speed_disc)

        road_option_features = self.road_option_embedding(metrics[:, 3].long())

        metrics_transformed = torch.cat([target_x_features, target_y_features, speed_features, road_option_features], dim=1)

        return metrics_transformed, metrics_transformed

class ProcessAction(nn.Module):
    def __init__(self, action_shape):
        super(ProcessAction, self).__init__()
        self.output_dim = action_shape

    def forward(self, action):
        action_transformed = action.clone()
        action_transformed.requires_grad = True

        return action_transformed, action_transformed