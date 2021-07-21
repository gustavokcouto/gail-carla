import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd

from common.running_mean_std import RunningMeanStd
from tools.utils import init
from tools.model import Flatten
import torch.optim as optim


class Discriminator(nn.Module):
    def __init__(self, state_shape, metrics_space, action_space, hidden_dim, device, lr, eps, betas, ct_lambda, max_grad_norm=None):
        super(Discriminator, self).__init__()
        self.device = device
        C, H, W = state_shape

        self.main = nn.Sequential(
            nn.Conv2d(C, 32, 4, stride=2, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, 4, stride=2, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            Flatten(),
        ).to(device)

        for i in range(4):
            H = (H - 4)//2 + 1
            W = (W - 4)//2 + 1
        # Get image dim
        img_dim = 256*H*W

        self.trunk = nn.Sequential(
            nn.Linear(
                img_dim + metrics_space.shape[0] + action_space.shape[0], hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)).to(device)
        self.last_layer = nn.Sequential(nn.Linear(hidden_dim, 1)).to(device)

        self.main.train()
        self.trunk.train()
        self.last_layer.train()

        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(list(self.main.parameters())
                                    + list(self.trunk.parameters())
                                    + list(self.last_layer.parameters()),
                                    lr=lr, betas=betas, eps=eps)
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.ct_lambda = ct_lambda

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
        mixup_state = alpha_state * expert_state + \
            (1 - alpha_state) * policy_state
        mixup_state.requires_grad = True

        alpha = alpha.view(expert_state.size(0), 1)
        alpha_metrics = alpha.expand_as(
            expert_metrics).to(expert_metrics.device)
        mixup_metrics = alpha_metrics * expert_metrics + \
            (1 - alpha_metrics) * policy_metrics
        mixup_metrics.requires_grad = True

        alpha_action = alpha.expand_as(expert_action).to(expert_action.device)
        mixup_action = alpha_action * expert_action + \
            (1 - alpha_action) * policy_action
        mixup_action.requires_grad = True

        mixup_state_features = self.main(mixup_state)

        mixup_data = torch.cat(
            [mixup_state_features, mixup_metrics, mixup_action], dim=1)

        disc_t = self.trunk(mixup_data)
        disc = self.last_layer(disc_t)
        ones = torch.ones(disc.size()).to(disc.device)

        grad = autograd.grad(
            outputs=disc,
            inputs=(mixup_state, mixup_metrics, mixup_action),
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        grad = grad.view(grad.size(0), -1)

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def compute_consistency_term(self,
                                 expert_state,
                                 expert_metrics,
                                 expert_action,
                                 m_tag=0):
        exp_state = self.main(expert_state)
        disc_t1 = self.trunk(
            torch.cat([exp_state, expert_metrics, expert_action], dim=1))
        disc1 = self.last_layer(disc_t1)

        exp_state = self.main(expert_state)
        disc_t2 = self.trunk(
            torch.cat([exp_state, expert_metrics, expert_action], dim=1))
        disc2 = self.last_layer(disc_t2)

        consistency_term = (disc1 - disc2).norm(2, dim=1) + 0.1 * \
            (disc_t1 - disc_t2).norm(2, dim=1) - m_tag
        return consistency_term.mean()

    def update(self, expert_loader, rollouts):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        expert_ac_loss = 0
        policy_ac_loss = 0
        g_loss = 0.0
        gp = 0.0
        ct = 0.0
        n = 0
        policy_reward = 0
        expert_reward = 0
        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):
            policy_state, policy_metrics, policy_action = policy_batch[
                0], policy_batch[1], policy_batch[2]
            policy_state = policy_state.to(self.device)
            policy_metrics = policy_metrics.to(self.device)
            policy_action = policy_action.to(self.device)

            pol_state = self.main(policy_state)
            policy_dt = self.trunk(
                torch.cat([pol_state, policy_metrics, policy_action], dim=1))
            policy_d = self.last_layer(policy_dt)
            policy_reward += policy_d.sum().item()

            expert_state, expert_metrics, expert_action = expert_batch

            expert_state = expert_state.to(self.device)
            expert_metrics = expert_metrics.to(self.device)
            expert_action = expert_action.to(self.device)
            exp_state = self.main(expert_state)
            expert_dt = self.trunk(
                torch.cat([exp_state, expert_metrics, expert_action], dim=1))
            expert_d = self.last_layer(expert_dt)
            expert_reward += expert_d.sum().item()

            # expert_loss = F.binary_cross_entropy_with_logits(
            #     expert_d,
            #     torch.ones(expert_d.size()).to(self.device))
            # policy_loss = F.binary_cross_entropy_with_logits(
            #     policy_d,
            #     torch.zeros(policy_d.size()).to(self.device))
            expert_loss = torch.mean(torch.tanh(expert_d)).to(self.device)
            policy_loss = torch.mean(torch.tanh(policy_d)).to(self.device)

            n_samples = policy_state.shape[0]
            expert_ac_loss += (expert_loss).item() * n_samples
            policy_ac_loss += (policy_loss).item() * n_samples
            # gail_loss = expert_loss + policy_loss
            wd = expert_loss - policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_metrics, expert_action,
                                             policy_state, policy_metrics, policy_action)
            cons_term = self.compute_consistency_term(expert_state, expert_metrics, expert_action)

            # loss += (gail_loss + grad_pen).item()
            loss += (-wd + grad_pen + self.ct_lambda * cons_term).item() * n_samples
            g_loss += (wd).item() * n_samples
            gp += (grad_pen).item() * n_samples
            ct += (cons_term).item() * n_samples
            n += n_samples

            self.optimizer.zero_grad()
            # (gail_loss + grad_pen).backward()
            (-wd + grad_pen + self.ct_lambda * cons_term).backward()
            nn.utils.clip_grad_norm_(
                self.main.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(
                self.trunk.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(
                self.last_layer.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return loss / n, policy_reward/n, expert_reward/n, g_loss/n, gp/n, ct/n, expert_ac_loss / n, policy_ac_loss / n

    def compute_loss(self, expert_loader, rollouts, batch_size=None):
        with torch.no_grad():
            policy_data_generator = rollouts.feed_forward_generator(
                None, mini_batch_size=expert_loader.batch_size, batch_size=batch_size)
            total_loss = 0
            policy_reward = 0
            expert_reward = 0

            n = 0
            for expert_batch, policy_batch in zip(expert_loader,
                                                  policy_data_generator):
                policy_state, policy_metrics, policy_action = policy_batch[
                    0], policy_batch[1], policy_batch[2]
                policy_state = policy_state.to(self.device)
                policy_metrics = policy_metrics.to(self.device)
                policy_action = policy_action.to(self.device)

                pol_state = self.main(policy_state)
                policy_dt = self.trunk(
                    torch.cat([pol_state, policy_metrics, policy_action], dim=1))
                policy_d = self.last_layer(policy_dt)

                expert_state, expert_metrics, expert_action = expert_batch
                expert_state = expert_state.to(self.device)
                expert_metrics = expert_metrics.to(self.device)
                expert_action = expert_action.to(self.device)
                exp_state = self.main(expert_state)
                expert_dt = self.trunk(
                    torch.cat([exp_state, expert_metrics, expert_action], dim=1))
                expert_d = self.last_layer(expert_dt)
                expert_loss = torch.tanh(expert_d)
                policy_loss = torch.tanh(policy_d)
                expert_reward += expert_loss.sum().item()
                policy_reward += policy_loss.sum().item()
                wd = expert_loss - policy_loss
                total_loss += wd.sum().item()
                n += policy_state.shape[0]

            if n == 0:
                return total_loss, expert_reward, policy_reward

            return total_loss/n, expert_reward/n, policy_reward/n

    def predict_reward(self, state, metrics, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()

            stat = self.main(state)
            dt = self.trunk(torch.cat([stat, metrics, action], dim=1))
            d = self.last_layer(dt)
            s = torch.sigmoid(d)
            reward = -(1 - s).log()
            reward = reward.cpu()

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20, start=0):
        all_trajectories = torch.load(file_name)

        perm = torch.randperm(all_trajectories['states'].size(0))
        # idx = perm[:num_trajectories]
        idx = np.arange(num_trajectories) + start

        self.trajectories = {}

        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories,)).long()

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
