class BehaviourClone():
    def __init__(self,
                 actor_critic,
                 lr=None,
                 eps=None,
                 expert_loader,
                 rollouts,
                 obsfilt=None):

        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        for expert_batch, policy_batch in zip(expert_loader,
                                              policy_data_generator):

            exp_obs_batch, expert_action = expert_batch
            exp_obs_batch = obsfilt(exp_obs_batch.numpy(), update=False)
            exp_obs_batch = torch.FloatTensor(exp_obs_batch).to(self.device)
            expert_action = expert_action.to(self.device)

            # Reshape to do in a single forward pass for all steps
            value, action, action_log_prob, recurrent_hidden_states _ = self.actor_critic.act(
                exp_obs_batch, recurrent_hidden_states_batch, masks_batch)

            self.optimizer.zero_grad()
            # (gail_loss + grad_pen).backward()
            (action - actions_batch).backward()
            self.optimizer.step()
