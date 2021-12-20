import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, metrics_shape, action_shape, radius=1):
        self.obs = torch.zeros(num_steps + 1, radius, num_processes, *obs_shape)
        self.metrics = torch.zeros(num_steps + 1, radius, num_processes, *metrics_shape)
        self.rewards = torch.zeros(num_steps, radius, num_processes, 1)
        self.gail_rewards = torch.zeros(num_steps, radius, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, radius, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, radius, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, radius, num_processes, 1)
        self.actions = torch.zeros(num_steps, radius, num_processes, *action_shape)
        self.masks = torch.ones(num_steps + 1, radius, num_processes, 1)
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.radius = radius
        self.step = 0
        self.iter = 0
        self.full = False

    def insert(self, obs, metrics, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1][self.iter].copy_(obs)
        self.metrics[self.step + 1][self.iter].copy_(metrics)
        self.actions[self.step][self.iter].copy_(actions)
        self.action_log_probs[self.step][self.iter].copy_(action_log_probs)
        self.value_preds[self.step][self.iter].copy_(value_preds)
        self.rewards[self.step][self.iter].copy_(rewards)
        self.masks[self.step + 1][self.iter].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        last_iter = self.iter
        self.iter = (self.iter + 1) % self.radius
        if self.iter == 0:
            self.full = True

        self.obs[0][self.iter].copy_(self.obs[-1][last_iter])
        self.metrics[0][self.iter].copy_(self.metrics[-1][last_iter])
        self.masks[0][self.iter].copy_(self.masks[-1][last_iter])
        self.step = 0

    def compute_returns(self,
                        gamma,
                        gae_lambda):
        gail_coef = 1.0
        env_coef = 0.0
        for iter in range(self.radius):
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = gail_coef * self.gail_rewards[step][iter] \
                    + env_coef * self.rewards[step][iter] \
                    + gamma * self.value_preds[step + 1][iter] \
                    * self.masks[step + 1][iter] - self.value_preds[step][iter]
                gae = delta + gamma * gae_lambda \
                    * self.masks[step + 1][iter] * gae
                self.returns[step][iter] = gae + self.value_preds[step][iter]

    def feed_forward_generator(self,
                               advantages,
                               mini_batch_size,
                               batch_size=None,
                               only_last_cycle=False):
        if only_last_cycle:
            radius = 1

            iter_init = self.iter
            iter_end = self.iter + 1

        else:
            radius = self.radius

            iter_init = 0
            iter_end = self.radius

        if batch_size is None:
            batch_size = radius * self.num_processes * self.num_steps

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1,iter_init:iter_end].reshape(-1, *self.obs.size()[3:])[indices]
            metrics_batch = self.metrics[:-1,iter_init:iter_end].reshape(-1, self.metrics.size(-1))[indices]
            actions_batch = self.actions[:,iter_init:iter_end].reshape(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1,iter_init:iter_end].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1,iter_init:iter_end].reshape(-1, 1)[indices]
            masks_batch = self.masks[:-1,iter_init:iter_end].reshape(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs[:,iter_init:iter_end].reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[:,iter_init:iter_end].reshape(-1, 1)[indices]

            yield obs_batch, metrics_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ