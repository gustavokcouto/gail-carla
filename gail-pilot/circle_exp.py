import math
import torch
import numpy as np
from circle_env import CircleEnv


def circle_expert(obs):
    center_x = 0.0
    center_y = 0.3
    pos_x = obs[-2]
    pos_y = obs[-1]
    action_angle = math.atan2(pos_y - center_y, pos_x - center_x) + math.pi / 2
    radius_factor = (
        (center_x ** 2 + center_y ** 2) ** 0.5
        - ((pos_x - center_x) ** 2 + (pos_y - center_y) ** 2) ** 0.5
    ) / (center_x ** 2 + center_y ** 2) ** 0.5
    action_angle -= 3 * radius_factor

    return [math.cos(action_angle), math.sin(action_angle)]


def gen_trajectories(file_path=''):
    # Instantiate the env
    env = CircleEnv()

    # Test the trained agent
    n_episodes = 20
    n_steps = 1000
    obs = env.reset()
    states = []
    actions = []
    rewards = []

    lens = []
    for _ in range(n_episodes):
        states_ep = []
        actions_ep = []
        rewards_ep = []
        obs = env.reset()
        states_ep.append(obs)
        for step in range(n_steps):
            action = circle_expert(obs)
            actions_ep.append(action)
            obs, reward, done, _ = env.step(action)
            rewards_ep.append(reward)
            if done:
                break
            states_ep.append(obs)
        states.append(states_ep)
        actions.append(actions_ep)
        rewards.append(rewards_ep)
        lens.append(step + 1)

    states = torch.as_tensor(states).float()
    actions = torch.as_tensor(actions).float()
    lens = torch.as_tensor(lens).long()
    rewards = torch.as_tensor(rewards).long()
    data = {
        'states': states,
        'actions': actions,
        'lengths': lens,
        'rewards': rewards
    }
    if file_path:
        torch.save(data, file_path)

if __name__ == "__main__":
    gen_trajectories('gail_experts/trajs_ant.pt')
    pass