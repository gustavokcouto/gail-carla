import torch
from algo.wdgail import ExpertDataset
from tools.model import Policy
import torch.optim as optim
from gym import spaces
import numpy as np
from carla_env import CarlaEnv
import tqdm
from tensorboardX import SummaryWriter
import os
import shutil


log_save_path = './runs'
if os.path.exists(log_save_path):
    shutil.rmtree(log_save_path)
writer = SummaryWriter(log_save_path)

file_name = "gail_experts/trajs_carla.pt"

expert_loader = torch.utils.data.DataLoader(
    ExpertDataset(
    file_name,
    num_trajectories=4,
    subsample_frequency=1),
    batch_size=512,
    shuffle=True,
    drop_last=True
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

action_space = spaces.Box(low=-10, high=10,
                                       shape=(1,), dtype=np.float32)

observation_space = spaces.Box(low=0, high=255,
                                        shape=(3,144,256), dtype=np.uint8)

metrics_space = spaces.Box(low=-100, high=100,
                                        shape=(2,), dtype=np.float32)

actor_critic = Policy(observation_space.shape, metrics_space, action_space)
actor_critic.to(device)

optimizer = optim.Adam(actor_critic.parameters(), lr=3e-4, eps=1e-5)

episodes = 100000
carla_env = CarlaEnv()
eval_step = 1000
ent_weight = 1e-3
time_step = 0
for episode in tqdm.tqdm(range(episodes)):
    for expert_batch in expert_loader:
        exp_obs_batch, exp_metrics_batch, expert_action_batch = expert_batch
        exp_obs_batch = exp_obs_batch.to(device)
        exp_metrics_batch = exp_metrics_batch.to(device)
        expert_action_batch = expert_action_batch.to(device)

        # Reshape to do in a single forward pass for all steps
        done, action_log_probs, entropy = actor_critic.evaluate_actions(exp_obs_batch, exp_metrics_batch, None, expert_action_batch)

        log_prob = action_log_probs.mean()
        loss = -log_prob + ent_weight * entropy

        writer.add_scalar('loss', loss, time_step)
        time_step += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % eval_step == 0 and episode != 0:
        steps = 1000
        obs, metrics = carla_env.reset()
        episode_done = False
        for step in range(steps):
            state = torch.FloatTensor([obs]).to(device)
            metrics = torch.FloatTensor([metrics]).to(device)
            _, action, _ = actor_critic.act(state, metrics, None, deterministic=True)
            obs, metrics, _, done, _ = carla_env.step(action.cpu().detach().numpy()[0])
            if done and not episode_done:
                writer.add_scalar('time_steps', step, time_step)
                episode_done = True

while True:
    steps = 1000
    obs, metrics = carla_env.reset()
    for _ in range(steps):
        state = torch.FloatTensor([obs]).to(device)
        metrics = torch.FloatTensor([metrics]).to(device)
        _, action, _ = actor_critic.act(state, metrics, None, deterministic=True)
        obs, metrics, _, done, _ = carla_env.step(action.cpu().detach().numpy()[0])
        if done:
            break