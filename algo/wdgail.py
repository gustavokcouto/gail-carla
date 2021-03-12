import itertools

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from common.running_mean_std import RunningMeanStd
from tools.utils import init
from tools.model import Flatten


class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, reward_type, update_rms, cliprew_down=-10.0, cliprew_up=10.0):
        super(Discriminator, self).__init__()
        self.cliprew_down = cliprew_down
        self.cliprew_up = cliprew_up
        self.device = device
        self.reward_type = reward_type
        self.update_rms = update_rms

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.state_encoder = nn.Sequential(
            init_(nn.Conv2d(3, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, 4, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 16, 3, stride=2)), nn.ReLU(), Flatten()
        ).to(device)

        self.trunk = nn.Sequential(
            nn.Linear(input_dim + 16 * 3 * 7, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()
        self.state_encoder.train()

        disc_params = itertools.chain(self.trunk.parameters(), self.state_encoder.parameters())
        self.optimizer = torch.optim.Adam(disc_params)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_metrics,
                         expert_action,
                         policy_state,
                         policy_metrics,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1, 1, 1)

        alpha_state = alpha.expand_as(expert_state).to(expert_state.device)
        mixup_state = alpha_state * expert_state + (1 - alpha_state) * policy_state
        mixup_state.requires_grad = True

        alpha = alpha.view(expert_state.size(0), 1)

        alpha_metrics = alpha.expand_as(expert_metrics).to(expert_metrics.device)
        mixup_metrics = alpha_metrics * expert_metrics + (1 - alpha_metrics) * policy_metrics
        mixup_metrics.requires_grad = True

        alpha_action = alpha.expand_as(expert_action).to(expert_action.device)
        mixup_action = alpha_action * expert_action + (1 - alpha_action) * policy_action
        mixup_action.requires_grad = True

        mixup_state_features = self.state_encoder(mixup_state)
        mixup_data = torch.cat([mixup_state_features, mixup_metrics, mixup_action], dim=1)
        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None, metricsfilt=None):
        self.trunk.train()
        self.state_encoder.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        g_loss =0.0
        gp =0.0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_metrics, policy_action = policy_batch[0], policy_batch[1], policy_batch[3]
            policy_state_features = self.state_encoder(policy_state)
            policy_d = self.trunk(
                torch.cat([policy_state_features, policy_metrics, policy_action], dim=1))

            expert_state, expert_metrics, expert_action = expert_batch

            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)

            expert_metrics = metricsfilt(expert_metrics.numpy(), update=False)
            expert_metrics = torch.FloatTensor(expert_metrics).to(self.device)

            expert_action = expert_action.to(self.device)
            expert_state_features = self.state_encoder(expert_state)
            expert_d = self.trunk(
                torch.cat([expert_state_features, expert_metrics, expert_action], dim=1))

            expert_loss = torch.mean(torch.tanh(expert_d)).to(self.device)
            policy_loss = torch.mean(torch.tanh(policy_d)).to(self.device)

            wd = expert_loss - policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_metrics, expert_action,
                                             policy_state, policy_metrics, policy_action)

            loss += (-wd + grad_pen).item()
            g_loss += (wd).item()
            gp += (grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (-wd + grad_pen).backward()
            self.optimizer.step()

        return g_loss/n, gp/n, 0.0, loss / n

    def predict_reward(self, state, metrics, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.trunk.eval()
            self.state_encoder.eval()
            state_features = self.state_encoder(state)
            d = self.trunk(torch.cat([state_features, metrics, action], dim=1))
            if self.reward_type == 0:
                s = torch.exp(d)
                reward = s
            elif self.reward_type == 1:
                s = torch.sigmoid(d)
                reward = - (1 - s).log()
            elif self.reward_type == 2:
                s = torch.sigmoid(d)
                reward = s
            elif self.reward_type == 3:
                s = torch.sigmoid(d)
                reward = s.exp()
            elif self.reward_type == 4:
                reward = d
            elif self.reward_type == 5:
                s = torch.sigmoid(d)
                reward = s.log() - (1 - s).log()

            if self.returns is None:
                self.returns = reward.clone()

            if self.update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
                return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
            else:
                return reward

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
        all_trajectories = torch.load(file_name)

        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}

        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories,)).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}

        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []

        for j in range(self.length):

            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1

    def __len__(self):
        return self.length

    def __getitem__(self, j):
        traj_idx, i = self.get_idx[j]

        return self.trajectories['states'][traj_idx][i], self.trajectories[
            'metrics'][traj_idx][i], self.trajectories['actions'][traj_idx][i]