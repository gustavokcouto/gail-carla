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


class Discriminator(nn.Module):
    def __init__(self, state_shape, metrics_space, action_space, hidden_dim, device, clip=0.01):
        super(Discriminator, self).__init__()
        self.device = device
        C, H, W = state_shape

        self.main = nn.Sequential(
            nn.Conv2d(C, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(), Flatten(),
        ).to(device)
        self.clip = clip
        print("Using clip {}".format(self.clip))

        for i in range(4):
            H = (H - 4)//2 + 1
            W = (W - 4)//2 + 1
        # Get image dim
        img_dim = 256*H*W

        self.trunk = nn.Sequential(
            nn.Linear(img_dim + metrics_space.shape[0] + action_space.shape[0], hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.main.train()
        self.trunk.train()

        self.optimizer = torch.optim.Adam(list(self.main.parameters()) + list(self.trunk.parameters()))
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

        # Change state values
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

        mixup_state_features = self.main(mixup_state)

        mixup_data = torch.cat([mixup_state_features, mixup_metrics, mixup_action], dim=1)

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)

        grad = autograd.grad(
            outputs=disc,
            inputs=(mixup_state, mixup_metrics, mixup_action),
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()
        assert obsfilt is None

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        expert_ac_loss = 0
        policy_ac_loss = 0
        g_loss =0.0
        gp =0.0
        n = 0
        policy_rewards = np.array([])
        expert_rewards = np.array([])
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_metrics, policy_action = policy_batch[0], policy_batch[1], policy_batch[2]

            pol_state = self.main(policy_state)
            policy_d = self.trunk(
                torch.cat([pol_state, policy_metrics, policy_action], dim=1))
            policy_batch_reward = policy_d.detach().cpu().numpy().reshape(-1)
            policy_rewards = np.concatenate((policy_rewards, policy_batch_reward), axis=0)

            expert_state, expert_metrics, expert_action = expert_batch

            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_metrics = expert_metrics.to(self.device)
            expert_action = expert_action.to(self.device)
            exp_state = self.main(expert_state)
            expert_d = self.trunk(
                torch.cat([exp_state, expert_metrics, expert_action], dim=1))
            expert_batch_rewards = expert_d.detach().cpu().numpy().reshape(-1)
            expert_rewards = np.concatenate((expert_rewards, expert_batch_rewards), axis=0)

            expert_loss = torch.mean(torch.tanh(expert_d)).to(self.device)
            policy_loss = torch.mean(torch.tanh(policy_d)).to(self.device)

            expert_ac_loss += (expert_loss).item()
            policy_ac_loss += (policy_loss).item()
            gail_loss = expert_loss - policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_metrics, expert_action,
                                             policy_state, policy_metrics, policy_action)

            loss += (-gail_loss + grad_pen).item()
            g_loss += (gail_loss).item()
            gp += (grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (-gail_loss + grad_pen).backward()
            self.optimizer.step()

        return loss / n, policy_rewards.mean(), expert_rewards.mean(), g_loss/n, gp/n, expert_ac_loss / n, policy_ac_loss / n

    def predict_reward(self, state, metrics, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            acts = action

            stat = self.main(state)
            d = self.trunk(torch.cat([stat, metrics, acts], dim=1))
            s = torch.sigmoid(d)
            reward = -(1 - s).log()

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


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
