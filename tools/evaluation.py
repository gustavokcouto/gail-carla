import torch

from carla_env import CarlaEnv
from tools.model import Policy


if __name__ == "__main__":
    env = CarlaEnv(
        'localhost',
        2000,
        2400,
        'data/route_00.xml',
        eval=True
    )
    # network
    actor_critic = Policy(
        env.observation_space.shape,
        env.metrics_space,
        env.action_space,
        activation=None,
        std_dev=[{'logstd': [-2.0, -3.2]}],
        var_ent=False
    )
    actor_critic.set_epoch(1)

    device = torch.device('cuda:0')

    loaddata = torch.load('carla_actor_bc.pt')
    actor_critic.load_state_dict(loaddata)
    actor_critic.to(device)

    n_episodes = 10
    total_reward = 0
    for _ in range(n_episodes):
        obs, metrics = env.reset()
        done = False
        while not done:
            obs = torch.from_numpy(obs).float().to(device)
            obs = torch.stack([obs])
            metrics = torch.from_numpy(metrics).float().to(device)
            metrics = torch.stack([metrics])
            with torch.no_grad():
                value, actions, action_log_prob = actor_critic.act(
                    obs,
                    metrics,
                    deterministic=True
                )
            action = actions.cpu().numpy()[0]

            obs, metrics, _, done, info = env.step(action)
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                ep_reward = info['episode']['r']
                print('Episode reward: {}'.format(ep_reward))
        total_reward += ep_reward
    
    ep_mean_reward = total_reward / n_episodes
    print('Episodes mean reward: {}'.format(ep_mean_reward))
