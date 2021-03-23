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


def learn_bc(actor_critic, env, device, expert_loader):
    log_save_path = './runs'
    if os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    writer = SummaryWriter(log_save_path)

    optimizer = optim.Adam(actor_critic.parameters(), lr=3e-4, eps=1e-5)

    episodes = 2000
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
            loss = -log_prob - ent_weight * entropy

            writer.add_scalar('loss', loss, time_step)
            time_step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % eval_step == 0 and episode != 0:
            steps = 1000
            obs, metrics = env.reset()
            episode_done = False
            for step in range(steps):
                state = torch.FloatTensor([obs]).to(device)
                metrics = torch.FloatTensor([metrics]).to(device)
                _, action, _ = actor_critic.act(state, metrics, None, deterministic=True)
                obs, metrics, _, done, _ = env.step(action.cpu().detach().numpy()[0])
                if done and not episode_done:
                    writer.add_scalar('time_steps', step, time_step)
                    episode_done = True