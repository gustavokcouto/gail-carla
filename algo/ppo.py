import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 mini_batch_size,
                 value_loss_coef,
                 device,
                 lr=None,
                 eps=None,
                 betas=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 gamma=None,
                 decay=None,
                 act_space=None,
                 ):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.act_space = act_space

        self.value_loss_coef = value_loss_coef

        self.device = device

        self.gamma = gamma
        self.decay = decay

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps, betas=betas)

    def update(self, rollouts, expert_dataset=None):
        # Expert dataset in case the BC update is required
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        gail_action_loss_epoch = 0
        dist_entropy_epoch = 0
        bc_loss_epoch = 0
        steer_std_epoch = 0
        throttle_std_epoch = 0

        n_updates = 0
        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.mini_batch_size)

            for sample in data_generator:
                obs_batch, metrics_batch, actions_batch, \
                   value_preds_batch, return_batch, _, old_action_log_probs_batch, \
                        adv_targ = sample
                obs_batch = obs_batch.to(self.device)
                metrics_batch = metrics_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)
                value_preds_batch = value_preds_batch.to(self.device)
                return_batch = return_batch.to(self.device)
                old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
                adv_targ = adv_targ.to(self.device)

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, steer_std, throttle_std = self.actor_critic.evaluate_actions(
                    obs_batch, metrics_batch, actions_batch, gail=True)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                gail_action_loss_epoch += action_loss.item()
                # Expert dataset
                if expert_dataset:
                    for exp_state, exp_metrics, exp_action in expert_dataset:
                        exp_state = Variable(exp_state).to(action_loss.device)
                        exp_metrics = Variable(exp_metrics).to(action_loss.device)
                        exp_action = Variable(exp_action).to(action_loss.device)
                        # Get BC loss
                        _, alogprobs, _, _, _ = self.actor_critic.evaluate_actions(exp_state, exp_metrics, exp_action)
                        bcloss = -alogprobs.mean()

                        bc_loss_epoch += bcloss.item()

                        # action loss is weighted sum
                        action_loss = self.gamma * bcloss + (1 - self.gamma) * action_loss
                        # Multiply this coeff with decay factor
                        break

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                steer_std_epoch += steer_std.item()
                throttle_std_epoch += throttle_std.item()
                n_updates += 1

        value_loss_epoch /= n_updates
        action_loss_epoch /= n_updates
        dist_entropy_epoch /= n_updates
        bc_loss_epoch /= n_updates
        gail_action_loss_epoch /= n_updates
        steer_std_epoch /= n_updates
        throttle_std_epoch /= n_updates

        if self.gamma is not None:
            self.gamma *= self.decay

        
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, bc_loss_epoch, \
            gail_action_loss_epoch, self.gamma, steer_std_epoch,  throttle_std_epoch