import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler



class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, metrics_shape, action_shape):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.metrics = torch.zeros(num_steps + 1, num_processes, *metrics_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.gail_rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, *action_shape)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.step = 0

    def insert(self, obs, metrics, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.metrics[self.step + 1].copy_(metrics)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps
        for i in range(self.num_processes):
            n_steps = metrics[i][4].long()
            if n_steps > 0:
                n_start = max(0, self.step - n_steps)
                self.metrics[n_start:self.step, i, 5:] = metrics[i][5:]
                self.metrics[self.step, i] = self.metrics[0, i]
    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.metrics[0].copy_(self.metrics[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self,
                        gamma,
                        gae_lambda):
        gae = 0
        gail_coef = 1.0
        env_coef = 0.0
        for step in reversed(range(self.rewards.size(0))):
            delta = gail_coef * self.gail_rewards[step] \
                + env_coef * self.rewards[step] \
                + gamma * self.value_preds[step + 1] \
                * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda \
                * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self,
                               advantages,
                               mini_batch_size,
                               batch_size=None,
                               only_last_cycle=False):
        if batch_size is None:
            batch_size = self.num_processes * self.num_steps

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            metrics_batch = self.metrics[:-1].view(-1, *self.metrics.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, metrics_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ