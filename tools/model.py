import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import init
from torchvision.models.resnet import resnet18, resnet34
from torchvision import transforms


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
        self.acc_exploration_dist = [
            torch.FloatTensor([0.0, 0.0]),  # 'void'
            torch.FloatTensor([1.0, 2.5]),  # 'go'
            torch.FloatTensor([1.5, 1.0])  # 'stop'
        ]
        self.steer_exploration_dist = [
            torch.FloatTensor([0.0, 0.0]),  # 'void'
            torch.FloatTensor([1.0, 1.0]),  # 'turn'
            torch.FloatTensor([3.0, 3.0])  # 'straight'
        ]

    def act(self, obs, metrics, deterministic=False):
        value, output = self.base(obs, metrics)
        dist = torch.distributions.Beta(output[..., :2], output[..., 2:])
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        # action = torch.max(torch.min(action, self.max.to(action)), self.min.to(action))
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        action = self.unscale_action(action)

        return value, action, action_log_probs

    def set_epoch(self,epoch):
        self.base.set_epoch(epoch)

    def get_value(self, obs, metrics):
        value, _ = self.base(obs, metrics)
        return value

    def exploration_loss(self, dist, metrics):
        steer_exploration = metrics[:, 5].long()
        acc_exploration = metrics[:, 6].long()
        alpha = dist.concentration1.detach().clone()
        beta = dist.concentration0.detach().clone()
        for i in range(len(alpha)):
            if steer_exploration[i] > 0:
                beta[i][0] = self.steer_exploration_dist[steer_exploration[i]][0]
                alpha[i][0] = self.steer_exploration_dist[steer_exploration[i]][1]
            if acc_exploration[i] > 0:
                beta[i][1] = self.acc_exploration_dist[acc_exploration[i]][0]
                alpha[i][1] = self.acc_exploration_dist[acc_exploration[i]][1]
        dist_exp = torch.distributions.Beta(alpha, beta)
        exploration_loss = torch.distributions.kl_divergence(dist, dist_exp)
        return torch.mean(exploration_loss)

    def evaluate_actions(self, obs, metrics, action):
        value, output = self.base(obs, metrics)
        dist = torch.distributions.Beta(output[..., :2], output[..., 2:])
        exploration_loss = self.exploration_loss(dist, metrics)
        action = self.scale_action(action)
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()
        steer_std = dist.entropy()[..., 0].sum(-1).mean()
        throttle_std = dist.entropy()[..., 1].sum(-1).mean()
        return value, action_log_probs, dist_entropy, exploration_loss, steer_std, throttle_std

    def scale_action(self, action, eps=1e-7):
        action[..., 0] = (action[..., 0] + 1) / 2
        action = torch.clamp(action, eps, 1-eps)
        return action

    def unscale_action(self, action):
        action[..., 0] = action[..., 0] * 2 - 1
        return action


class CNNBase(nn.Module):
    def __init__(self, obs_shape, metrics_space, num_outputs, activation, logstd, multi_head):
        super(CNNBase, self).__init__()

        self.obs_processor = ProcessObsFeatures(obs_shape)

        self.metrics_processor = ProcessMetrics(metrics_space.shape[0])
        hidden_size = 512
        self.body = NNBody(self.obs_processor.output_dim +
                           self.metrics_processor.output_dim, hidden_size)
        self.head = NNHead(hidden_size, 2 * num_outputs, value=True)

        self.activation = nn.Softplus()

    def forward(self, obs, metrics):
        obs_features, _ = self.obs_processor(obs)
        metrics_features, _ = self.metrics_processor(metrics)
        cat_features = torch.cat([obs_features, metrics_features], dim=1)
        body_features = self.body(cat_features)
        road_options = metrics[:, 3].long()
        road_options -= 1
        critic, output = self.head(body_features)

        output = self.activation(output)
        return critic, output


class NNBody(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNBody, self).__init__()
        hidden_size = 512
        self.body = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(0.2),
        )
        self.output_size = output_size

    def forward(self, input):
        output = self.body(input)
        return output


class NNHead(nn.Module):
    def __init__(self, input_size, output_size, value=False):
        super(NNHead, self).__init__()
        hidden_size = 256
        self.value = value

        if self.value:
            output_size += 1

        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, input):
        output = self.head(input)
        if self.value:
            return output[..., 0].unsqueeze(dim=1), output[..., 1:]
        else:
            return output


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
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406,],
                                              std=[0.229, 0.224, 0.225])

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

# class ProcessMetrics(nn.Module):
#     def __init__(self, metrics_shape):
#         super(ProcessMetrics, self).__init__()

#         speed_shape = 1
#         input_size = speed_shape
#         hidden_size = 128
#         self.output_dim = 128

#         self.linear = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.LeakyReLU(0.2),
#             nn.Linear(hidden_size, self.output_dim),
#             nn.LeakyReLU(0.2),
#         )

#     def forward(self, metrics):
#         # metrics composition [target[0], target[1], speed, int(road_option)]

#         metrics_copy = metrics.clone()

#         # max speed of 60m/s or 216km/h
#         metrics_transformed = 0.1 * metrics_copy[:, 2].unsqueeze(dim=1)
#         metrics_transformed.requires_grad = True
        
#         metrics_features = self.linear(metrics_transformed)

#         return metrics_features, metrics_transformed
