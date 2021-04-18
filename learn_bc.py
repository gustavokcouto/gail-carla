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


def learn_bc(actor_critic, device, expert_loader):
    log_save_path = './runs'
    if os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    writer = SummaryWriter(log_save_path)

    optimizer = optim.Adam(actor_critic.parameters(), lr=3e-4, eps=1e-5)

    episodes = 2000
    eval_step = 1000
    ent_weight = 0
    time_step = 0
    i_eval = 0
    for episode in tqdm.tqdm(range(episodes)):
        for expert_batch in expert_loader:
            exp_obs_batch, exp_metrics_batch, expert_action_batch = expert_batch
            exp_obs_batch = exp_obs_batch.to(device)
            exp_metrics_batch = exp_metrics_batch.to(device)
            expert_action_batch = expert_action_batch.to(device)

            # Reshape to do in a single forward pass for all steps
            _, action_log_probs, entropy = actor_critic.evaluate_actions(exp_obs_batch, exp_metrics_batch, expert_action_batch)

            log_prob = action_log_probs.mean()
            loss = -log_prob - ent_weight * entropy

            writer.add_scalar('loss', loss, time_step)
            time_step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % eval_step == 0 and episode != 0:
            torch.save(actor_critic.state_dict(), 'carla_actor_eval{}.pt'.format(i_eval))
            i_eval += 1


if __name__ == '__main__':
    env = CarlaEnv()
    # network
    actor_critic = Policy(
        env.observation_space.shape,
        env.metrics_space,
        env.action_space,
        activation=None)
    
    device = torch.device('cuda:0')

    actor_critic.to(device)

    file_name = 'gail_experts/trajs_carla.pt'

    gail_train_loader = torch.utils.data.DataLoader(
        ExpertDataset(
        file_name, num_trajectories=2, subsample_frequency=1),
        batch_size=128,
        shuffle=True,
        drop_last=True)
    
    learn_bc(actor_critic, device, gail_train_loader)