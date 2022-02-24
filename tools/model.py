import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import init
from torchvision.models.resnet import resnet18, resnet34
from torchvision import transforms
from tools.distributions import DiagGaussianDistribution

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, metrics_space, action_space, activation, logstd, multi_head):
        super(Policy, self).__init__()

        num_outputs = action_space.shape[0]
        self.base = CNNBase(obs_shape, metrics_space, num_outputs, activation, logstd, multi_head)

        self.max = torch.Tensor([1, 1])
        self.min = torch.Tensor([-1, 0])

    def act(self, obs, metrics, deterministic=False):
        value, dist = self.base(obs, metrics)
        action = dist.get_actions(deterministic=deterministic)

        action_log_probs = dist.distribution.log_prob(action).sum(-1, keepdim=True)
        action = self.unscale_action(action)
        return value, action, action_log_probs

    def set_epoch(self,epoch):
        self.base.set_epoch(epoch)

    def get_value(self, obs, metrics):
        value, _ = self.base(obs, metrics)
        return value

    def evaluate_actions(self, obs, metrics, action):
        value, distribution = self.base(obs, metrics)
        dist = distribution.distribution
        action = self.scale_action(action)
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()
        steer_std = 0
        throttle_std = 0
        exploration_loss = 0
        return value, action_log_probs, dist_entropy, exploration_loss, steer_std, throttle_std

    def scale_action(self, action, eps=1e-7):
        # action[..., 0] = (action[..., 0] + 1) / 2
        # action = torch.clamp(action, eps, 1-eps)
        return action

    def unscale_action(self, action):
        # action[..., 0] = action[..., 0] * 2 - 1
        return action


class CNNBase(nn.Module):
    def __init__(self, obs_shape, metrics_space, num_outputs, activation, logstd, multi_head):
        super(CNNBase, self).__init__()

        self.obs_processor = ProcessObsFeatures(obs_shape)

        self.metrics_processor = ProcessMetrics(metrics_space.shape[0])
        hidden_size = 256
        self.body = NNBody(self.obs_processor.output_dim +
                           self.metrics_processor.output_dim, hidden_size)
        self.head = NNPolicyHead(hidden_size, hidden_size)
        self.action_dist = DiagGaussianDistribution()
        self.dist_mu, self.dist_sigma = self.action_dist.proba_distribution_net(hidden_size)
        self.logstd = torch.Tensor(logstd)

        self.activation = activation

    def forward(self, obs, metrics):
        obs_features, _ = self.obs_processor(obs)
        metrics_features, _ = self.metrics_processor(metrics)
        cat_features = torch.cat([obs_features, metrics_features], dim=1)
        body_features = self.body(cat_features)
        critic, head_output = self.head(body_features)
        mu = self.dist_mu(head_output)
        mu[..., 0] = torch.tanh(mu[..., 0])
        mu[..., 1] = torch.sigmoid(mu[..., 1])
        if isinstance(self.dist_sigma, nn.Parameter):
            sigma = self.dist_sigma
        else:
            sigma = self.dist_sigma(head_output)
        distribution = self.action_dist.proba_distribution(mu, sigma)
        return critic, distribution


class NNBody(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNBody, self).__init__()
        hidden_size = 512
        self.body = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(0.2),
        )
        self.output_size = output_size

    def forward(self, input):
        output = self.body(input)
        return output



class NNHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNHead, self).__init__()
        hidden_size = 256

        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, input):
        output = self.head(input)
        return output


class NNPolicyHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNPolicyHead, self).__init__()
        hidden_size = 256

        self.policy_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(0.2)
        )

        self.value_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input):
        output = self.policy_head(input)
        value = self.value_head(input)
        return value, output


class ProcessObsFeatures(nn.Module):
    def __init__(self, obs_shape):
        super(ProcessObsFeatures, self).__init__()
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

        for _ in range(4):
            H = (H - 4)//2 + 1
            W = (W - 4)//2 + 1

        # Get image dim
        self.output_dim = 256*H*W
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225])

    def forward(self, obs):
        # scale observation
        obs_transformed = obs.clone()
        obs_transformed.requires_grad = True
        obs_normalized = self.normalize(obs_transformed)
        obs_features = self.main(obs_normalized)

        return obs_features, obs_transformed


class ProcessMetrics(nn.Module):
    def __init__(self, metrics_shape):
        super(ProcessMetrics, self).__init__()

        road_option_embedding_dimension = 8
        max_road_options = 10
        self.road_option_embedding = nn.Embedding(max_road_options, road_option_embedding_dimension)
        # target x, y, r, theta
        target_shape = 4
        speed_shape = 1
        self.output_dim = target_shape + speed_shape + road_option_embedding_dimension

    def forward(self, metrics):
        # metrics composition [target[0], target[1], speed, int(road_option)]

        metrics_copy = metrics.clone().cpu().numpy()
        target_x = metrics_copy[:, 0]
        target_y = metrics_copy[:, 1]
        target_r = np.sqrt(target_x * target_x + target_y * target_y)
        target_theta = np.arctan2(target_y, target_x) 

        # scale target x and y by 1000
        metrics_target_x = 1000 * torch.from_numpy(target_x).float().unsqueeze(dim=1)
        metrics_target_y = 1000 * torch.from_numpy(target_y).float().unsqueeze(dim=1)

        # scale target radius by 1000
        metrics_target_r = 1000 * torch.from_numpy(target_r).float().unsqueeze(dim=1)

        # scale target theta by 0.3
        metrics_target_theta = 0.3 * torch.from_numpy(target_theta).float().unsqueeze(dim=1)

        # max speed of 60m/s or 216km/h
        speed = metrics_copy[:, 2]

        # scale speed by 0.1
        metrics_speed = 0.1 * torch.from_numpy(speed).float().unsqueeze(dim=1)

        road_options = metrics_copy[:, 3]
        road_options_tensor = torch.from_numpy(road_options).long().to(metrics.device)
        road_option_features = self.road_option_embedding(road_options_tensor)

        metrics_transformed = torch.cat([metrics_target_x, metrics_target_y, metrics_target_r, metrics_target_theta, metrics_speed], dim=1).clone().to(metrics.device)
        metrics_transformed.requires_grad = True

        metrics_transformed = torch.cat([metrics_transformed, road_option_features], dim=1)

        return metrics_transformed, metrics_transformed

class ProcessAction(nn.Module):
    def __init__(self, action_shape):
        super(ProcessAction, self).__init__()
        self.output_dim = action_shape

    def forward(self, action):
        action_transformed = action.clone()
        action_transformed.requires_grad = True

        return action_transformed, action_transformed


class ProcessObsFeaturesResnet(nn.Module):
    def __init__(self, obs_shape, resnet_34=False):
        super(ProcessObsFeaturesResnet, self).__init__()
        in_channels = obs_shape[0]
        self.main = resnet34(pretrained=True).eval()
        self.main.fc = nn.Identity()
        self.output_dim = 512 * 3
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, obs):
        # scale observation
        obs_transformed = obs.clone()
        with torch.no_grad():
            obs_left = self.normalize(obs_transformed[:, :3])
            obs_center = self.normalize(obs_transformed[:, 3:6])
            obs_right = self.normalize(obs_transformed[:, 6:])
            left_feat = self.main(obs_left)
            center_feat = self.main(obs_center)
            right_feat = self.main(obs_right)
            obs_features = torch.cat([left_feat, center_feat, right_feat], dim=1)

        return obs_features, obs_transformed