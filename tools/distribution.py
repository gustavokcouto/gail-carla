class DiagGaussianDistribution():
    def __init__(self, action_dim: int, dist_init=None):
        self.distribution = None
        self.action_dim = action_dim
        self.dist_init = dist_init

        self.log_std_max = 2
        self.log_std_min = -20

        # [mu, log_std], [0, 1]
        self.acc_exploration_dist = {
            'go': torch.FloatTensor([0.66, -3]),
            'stop': torch.FloatTensor([-0.66, -3])
        }
        self.steer_exploration_dist = {
            'turn': torch.FloatTensor([0.0, -1]),
            'straight': torch.FloatTensor([3.0, 3.0])
        }

        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Parameter]:
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        if self.action_dependent_std:
            log_std = nn.Linear(latent_dim, self.action_dim)
        else:
            log_std = nn.Parameter(-2.0*th.ones(self.action_dim), requires_grad=True)

        if self.dist_init is not None:
            # log_std.weight.data.fill_(0.01)
            # mean_actions.weight.data.fill_(0.01)
            # acc/steer
            mean_actions.bias.data[0] = self.dist_init[0][0]
            mean_actions.bias.data[1] = self.dist_init[1][0]
            if self.action_dependent_std:
                log_std.bias.data[0] = self.dist_init[0][1]
                log_std.bias.data[1] = self.dist_init[1][1]
            else:
                init_tensor = th.FloatTensor([self.dist_init[0][1], self.dist_init[1][1]])
                log_std = nn.Parameter(init_tensor, requires_grad=True)

        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "DiagGaussianDistribution":
        if self.action_dependent_std:
            log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy_loss(self) -> th.Tensor:
        entropy_loss = -1.0 * self.distribution.entropy()
        return th.mean(entropy_loss)

    def exploration_loss(self, exploration_suggests) -> th.Tensor:
        # [('stop'/'go'/None, 'turn'/'straight'/None)]
        # (batch_size, action_dim)
        mu = self.distribution.loc.detach().clone()
        sigma = self.distribution.scale.detach().clone()

        for i, (acc_suggest, steer_suggest) in enumerate(exploration_suggests):
            if acc_suggest != '':
                mu[i, 0] = self.acc_exploration_dist[acc_suggest][0]
                sigma[i, 0] = self.acc_exploration_dist[acc_suggest][1]
            if steer_suggest != '':
                mu[i, 1] = self.steer_exploration_dist[steer_suggest][0]
                sigma[i, 1] = self.steer_exploration_dist[steer_suggest][1]

        dist_ent = Normal(mu, sigma)

        exploration_loss = th.distributions.kl_divergence(dist_ent, self.distribution)
        return th.mean(exploration_loss)

    def sample(self) -> th.Tensor:
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()