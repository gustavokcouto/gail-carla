import torch

from carla_env import CarlaEnv
from tools.model import Policy


if __name__ == "__main__":
    env = CarlaEnv()
    # network
    actor_critic = Policy(
        env.observation_space.shape,
        env.metrics_space,
        env.action_space,
        activation=None)
    
    device = torch.device('cuda:0')

    loaddata = torch.load('carla_actor.pt')
    actor_critic.load_state_dict(loaddata)
    actor_critic.to(device)

    while True:
        obs, metrics = env.reset()
        while True:
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

            obs, metrics, rewards, done, infos = env.step(action)

            if done:
                break