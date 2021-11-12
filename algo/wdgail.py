import itertools

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

from tools.model import ProcessObsFeatures, ProcessObsFeaturesResnet, ProcessMetrics, ProcessAction, OutputLayers
import torch.optim as optim
from PIL import Image

from common.running_mean_std import RunningMeanStd


class Discriminator(nn.Module):
    def __init__(self, state_shape, metrics_space, action_space, hidden_dim, device, lr, eps, betas, max_grad_norm=None):
        super(Discriminator, self).__init__()
        self.device = device

        self.obs_processor = ProcessObsFeatures(state_shape, bias=False)
        self.metrics_processor = ProcessMetrics(metrics_space.shape[0])
        self.action_processor = ProcessAction(action_space.shape[0])

        input_size = self.obs_processor.output_dim + self.metrics_processor.output_dim + self.action_processor.output_dim
        output_size = 1
        self.trunk = OutputLayers(input_size, output_size, hidden_size=hidden_dim)

        self.train()

        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=lr, betas=betas, eps=eps)
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
        mixup_state = alpha_state * expert_state + \
            (1 - alpha_state) * policy_state

        alpha = alpha.view(expert_state.size(0), 1)
        alpha_metrics = alpha.expand_as(
            expert_metrics).to(expert_metrics.device)
        mixup_metrics = alpha_metrics * expert_metrics + \
            (1 - alpha_metrics) * policy_metrics

        alpha_action = alpha.expand_as(expert_action).to(expert_action.device)
        mixup_action = alpha_action * expert_action + \
            (1 - alpha_action) * policy_action

        mixup_state_features, mixup_state_transformed = self.obs_processor(mixup_state)
        mixup_metrics_features, mixup_metrics_transformed = self.metrics_processor(mixup_metrics)
        mixup_action_features, mixup_action_transformed = self.action_processor(mixup_action)
        
        mixup_data = torch.cat(
            [mixup_state_features, mixup_metrics_features, mixup_action_features], dim=1)

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)

        grad = autograd.grad(
            outputs=disc,
            inputs=(mixup_state_transformed, mixup_metrics_transformed, mixup_action_transformed),
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        grad = grad.view(grad.size(0), -1)

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        expert_ac_loss = 0
        policy_ac_loss = 0
        g_loss = 0.0
        gp = 0.0
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

            policy_state_features, _ = self.obs_processor(policy_state)
            policy_metrics_features, _ = self.metrics_processor(policy_metrics)
            policy_action_features, _ = self.action_processor(policy_action)

            policy_d = self.trunk(
                torch.cat([policy_state_features,  policy_metrics_features, policy_action_features], dim=1))
            policy_reward += policy_d.sum().item()

            expert_state, expert_metrics, expert_action = expert_batch

            expert_state = expert_state.to(self.device)
            expert_metrics = expert_metrics.to(self.device)
            expert_action = expert_action.to(self.device)

            expert_state_features, _ = self.obs_processor(expert_state)
            expert_metrics_features, _ = self.metrics_processor(expert_metrics)
            expert_action_features, _ = self.action_processor(expert_action)

            expert_d = self.trunk(
                torch.cat([expert_state_features, expert_metrics_features, expert_action_features], dim=1))
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

            # loss += (gail_loss + grad_pen).item()
            loss += (-wd + grad_pen).item() * n_samples
            g_loss += (wd).item() * n_samples
            gp += (grad_pen).item() * n_samples
            n += n_samples

            self.optimizer.zero_grad()
            # (gail_loss + grad_pen).backward()
            (-wd + grad_pen).backward()
            nn.utils.clip_grad_norm_(
                self.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return loss / n, policy_reward/n, expert_reward/n, g_loss/n, gp/n, expert_ac_loss / n, policy_ac_loss / n

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

                policy_state_features, _ = self.obs_processor(policy_state)
                policy_metrics_features, _ = self.metrics_processor(policy_metrics)
                policy_action_features, _ = self.action_processor(policy_action)

                policy_d = self.trunk(
                    torch.cat([policy_state_features, policy_metrics_features, policy_action_features], dim=1))

                expert_state, expert_metrics, expert_action = expert_batch
                expert_state = expert_state.to(self.device)
                expert_metrics = expert_metrics.to(self.device)
                expert_action = expert_action.to(self.device)

                expert_state_features, _ = self.obs_processor(expert_state)
                expert_metrics_features, _ = self.metrics_processor(expert_metrics)
                expert_action_features, _ = self.action_processor(expert_action)

                expert_d = self.trunk(
                    torch.cat([expert_state_features, expert_metrics_features, expert_action_features], dim=1))
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

            state_features, _ = self.obs_processor(state)
            metrics_features, _ = self.metrics_processor(metrics)
            action_features, _ = self.action_processor(action)

            d = self.trunk(
                torch.cat([state_features, metrics_features, action_features], dim=1))
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
        self.dataset_dir = Path(file_name).parent

        perm = torch.randperm(all_trajectories['actions'].size(0))
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
        self.actual_obs = [None for _ in range(self.length)]

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
        # Load only the first time, images in uint8 are supposed to be light
        if self.actual_obs[j] is None:
            rgb = Image.open(self.dataset_dir /
                            'episode_{:0>2d}/rgb/{:0>4d}.png'.format(traj_idx, i))
            rgb_left = Image.open(
                self.dataset_dir / 'episode_{:0>2d}/rgb_left/{:0>4d}.png'.format(traj_idx, i))
            rgb_right = Image.open(
                self.dataset_dir / 'episode_{:0>2d}/rgb_right/{:0>4d}.png'.format(traj_idx, i))
            rgb = np.transpose(rgb, (2, 0, 1))
            rgb_left = np.transpose(rgb_left, (2, 0, 1))
            rgb_right = np.transpose(rgb_right, (2, 0, 1))
            obs = np.concatenate((rgb, rgb_left, rgb_right))
            self.actual_obs[j] = obs
        else:
            obs = self.actual_obs[j]

        return torch.from_numpy(obs).float(), self.trajectories[
            'metrics'][traj_idx][i], self.trajectories['actions'][traj_idx][i]
