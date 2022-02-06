import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import init
from torchvision.models.resnet import ResNet, BasicBlock, resnet18, resnet34
from torchvision import transforms


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, metrics_space, action_space, activation, logstd, multi_head, resnet):
        super(Policy, self).__init__()

        num_outputs = action_space.shape[0]
        self.base = CNNBase(obs_shape, metrics_space, num_outputs,
                            activation, logstd, multi_head, resnet)

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

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def get_value(self, obs, metrics):
        value, _, _ = self.base(obs, metrics)
        return value

    def evaluate_actions(self, obs, metrics, action, gail=False):
        value, output, logstd = self.base(obs, metrics, gail=gail)
        dist = torch.distributions.Normal(output, logstd.exp())

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()
        steer_std = logstd[0, 0].detach().cpu()
        throttle_std = logstd[0, 1].detach().cpu()
        return value, action_log_probs, dist_entropy, steer_std, throttle_std


class CNNBase(nn.Module):
    def __init__(self, obs_shape, metrics_space, num_outputs, activation, logstd, multi_head, resnet):
        super(CNNBase, self).__init__()

        self.obs_processor = ProcessObsFeaturesResnet(
            obs_shape, resnet_34=True)

        self.metrics_processor = ProcessMetrics(metrics_space.shape[0])
        hidden_size = 256
        self.body = NNBody(self.obs_processor.output_dim +
                           self.metrics_processor.output_dim, hidden_size)
        self.head = NNHead(hidden_size, num_outputs, value=True)
        self.logstd = torch.Tensor(logstd)

        self.linear_params = list(self.metrics_processor.parameters()) + list(self.body.parameters()) + list(self.head.parameters())
        self.image_params = self.obs_processor.parameters()
        self.activation = activation

    def forward(self, obs, metrics, gail=False):
        if gail:
            with torch.no_grad():
                obs_features, _ = self.obs_processor(obs)
        else:
            obs_features, _ = self.obs_processor(obs)
        metrics_features, _ = self.metrics_processor(metrics)
        cat_features = torch.cat([obs_features, metrics_features], dim=1)
        nn_body = self.body(cat_features)
        road_options = metrics[:, 3].long()
        road_options -= 1
        critic, output = self.head(nn_body, road_options)

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
        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size)
        )
        self.value = value
        if self.value:
            self.value_branch = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, 1),
            )
    def forward(self, input, command):
        output = self.head(input)
        if self.value:
            value = self.value_branch(input)
            return value, output
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

    def forward(self, obs):
        # scale observation
        obs_transformed = obs.clone()
        obs_transformed.requires_grad = True
        obs_features = self.main(obs_transformed)

        return obs_features, obs_transformed


class ProcessMetrics(nn.Module):
    def __init__(self, metrics_shape, action_shape=None):
        super(ProcessMetrics, self).__init__()

        # target x, y, r, theta
        hidden_size = 128
        self.output_dim = hidden_size
        input_dim = 4
        if action_shape:
            self.process_action = True
            input_dim += action_shape
        else:
            self.process_action = False

        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2)
        )

    def forward(self, metrics, action=None):
        # metrics composition [target[0], target[1], speed, int(road_option)]

        # max speed of 60m/s or 216km/h
        speed = 0.1 * metrics[:, 2].unsqueeze(dim=1)

        road_option = metrics[:, 3].unsqueeze(dim=1)

        # scale target x and y by 1000
        target_x = 1000 * metrics[:, 0].unsqueeze(dim=1)
        target_y = 1000 * metrics[:, 1].unsqueeze(dim=1)

        metrics_transformed = torch.cat([speed, road_option, target_x, target_y], dim=1)
        if self.process_action:
            metrics_transformed = torch.cat([metrics_transformed, action], dim=1)

        metrics_transformed.requires_grad = True

        metrics_features = self.main(metrics_transformed)

        return metrics_features, metrics_transformed


class ProcessObsFeaturesResnet(nn.Module):
    def __init__(self, obs_shape, resnet_34=False):
        super(ProcessObsFeaturesResnet, self).__init__()
        in_channels = obs_shape[0]
        self.main = resnet18(pretrained=False).eval()
        old = self.main.conv1
        self.main.conv1 = torch.nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=old.bias)
        self.output_dim = 1000
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225])

    def forward(self, obs):
        # scale observation
        obs_transformed = obs.clone()
        obs_transformed.requires_grad = True

        obs_normalized = self.normalize(obs_transformed)
        resnet_features = self.main(obs_normalized)

        return resnet_features, obs_transformed
