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
    def __init__(self, obs_shape, metrics_space, num_outputs, activation, logstd, multi_head):
        super(CNNBase, self).__init__()

        self.obs_processor = ProcessObsFeatures(obs_shape)

        self.metrics_processor = ProcessMetrics(metrics_space.shape[0])
        hidden_size = 512
        self.body = NNBody(self.obs_processor.output_dim +
                           self.metrics_processor.output_dim, hidden_size)
        self.head = NNBranchedHead(hidden_size, num_outputs, value=True)
        self.logstd = torch.Tensor(logstd)

        self.activation = activation

    def forward(self, obs, metrics):
        obs_features, _ = self.obs_processor(obs)
        metrics_features, _ = self.metrics_processor(metrics)
        cat_features = torch.cat([obs_features, metrics_features], dim=1)
        body_features = self.body(cat_features)
        road_options = metrics[:, 3].long()
        road_options -= 1
        critic, output = self.head(body_features, road_options)

        if self.activation:
            output[..., 0] = torch.tanh(output[..., 0])
            output[..., 1] = torch.sigmoid(output[..., 1])
        zeros = torch.zeros(output.size()).to(output)
        logstd = self.logstd.to(output)
        logstd = logstd + zeros
        return critic, output, logstd


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


class NNBranchedHead(nn.Module):
    def __init__(self, input_size, output_size, value=False):
        super(NNBranchedHead, self).__init__()
        number_of_branches = 6
        branch_vector = []
        hidden_size = 256
        for i in range(number_of_branches):
            branch_head = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, output_size),
            )
            branch_vector.append(branch_head)
        self.branched_modules = nn.ModuleList(branch_vector)

        self.value = value
        if self.value:
            self.value_branch = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, input, command):
        branches_outputs = []
        for branch in self.branched_modules:
            branches_outputs.append(branch(input))

        branches_outputs = torch.stack(branches_outputs)
        command = command.type(torch.LongTensor)
        branch_number = torch.LongTensor(range(0, command.size(0)))
        output = branches_outputs[command, branch_number, :]
        if self.value:
            value = self.value_branch(input)
            return value, output
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
    def __init__(self, metrics_shape, command=False):
        super(ProcessMetrics, self).__init__()

        # target x, y, r, theta
        hidden_size = 128
        self.output_dim = hidden_size
        input_dim = 1
        self.command = command
        if self.command:
            road_option_embedding_dimension = 8
            max_road_options = 10
            self.road_option_embedding = nn.Embedding(max_road_options, road_option_embedding_dimension)
            input_dim += road_option_embedding_dimension

        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2)
        )

    def forward(self, metrics):
        # metrics composition [target[0], target[1], speed, int(road_option)]

        # max speed of 60m/s or 216km/h
        speed = 0.1 * metrics[:, 2].unsqueeze(dim=1)
        metrics_transformed = speed.clone()
        metrics_transformed.requires_grad = True

        if self.command:
            road_options = metrics[:, 3].long()
            road_option_features = self.road_option_embedding(road_options)
            metrics_transformed = torch.cat([metrics_transformed, road_option_features], dim=1)

        metrics_features = self.main(metrics_transformed)

        return metrics_features, metrics_transformed


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