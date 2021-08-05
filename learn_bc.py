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


def learn_bc(actor_critic, device, expert_loader, eval_loader):
    log_save_path = './runs'
    if os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    writer = SummaryWriter(log_save_path)

    optimizer = optim.Adam(actor_critic.parameters(), lr=3e-4, eps=1e-5)

    episodes = 500
    ent_weight = 0
    i_epoch = 0
    min_eval_loss = np.inf
    for _ in tqdm.tqdm(range(episodes)):
        total_loss = 0
        i_batch = 0
        for expert_batch in expert_loader:
            exp_obs_batch, exp_metrics_batch, expert_action_batch = expert_batch
            exp_obs_batch = exp_obs_batch.to(device)
            exp_metrics_batch = exp_metrics_batch.to(device)
            expert_action_batch = expert_action_batch.to(device)

            # Reshape to do in a single forward pass for all steps
            _, action_log_probs, entropy, _, _ = actor_critic.evaluate_actions(exp_obs_batch, exp_metrics_batch, expert_action_batch)

            log_prob = action_log_probs.mean()
            loss = -log_prob - ent_weight * entropy
            total_loss += loss
            i_batch += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_eval_loss = 0
        i_eval_batch = 0
        for eval_batch in eval_loader:
            eval_obs_batch, eval_metrics_batch, eval_action_batch = eval_batch
            eval_obs_batch = eval_obs_batch.to(device)
            eval_metrics_batch = eval_metrics_batch.to(device)
            eval_action_batch = eval_action_batch.to(device)

            # Reshape to do in a single forward pass for all steps
            with torch.no_grad():
                _, action_log_probs, entropy, _, _ = actor_critic.evaluate_actions(eval_obs_batch, eval_metrics_batch, eval_action_batch)

            log_prob = action_log_probs.mean()
            eval_loss = -log_prob - ent_weight * entropy
            total_eval_loss += eval_loss
            i_eval_batch += 1
        
        loss = total_loss / i_batch
        eval_loss = total_eval_loss / i_eval_batch
        writer.add_scalar('loss', loss, i_epoch)
        writer.add_scalar('eval_loss', eval_loss, i_epoch)
        i_epoch += 1

        if min_eval_loss > eval_loss:
            torch.save(actor_critic.state_dict(), 'carla_actor_bc.pt')


if __name__ == '__main__':
    env = CarlaEnv('localhost', 2000, 800, 'data/route_01.xml')
    # network
    actor_critic = Policy(
        env.observation_space.shape,
        env.metrics_space,
        env.action_space,
        activation=None,
        std_dev=[{'logstd': [-2.0, -3.2]}],
        var_ent=False)
    actor_critic.set_epoch(1)
    device = torch.device('cuda:0')

    actor_critic.to(device)
    route = 'route_01'
    file_name = 'gail_experts/{}/trajs_carla.pt'.format(route)

    gail_train_loader = torch.utils.data.DataLoader(
        ExpertDataset(
        file_name, num_trajectories=10, subsample_frequency=1),
        batch_size=128,
        shuffle=True,
        drop_last=True)
    
    gail_val_loader = torch.utils.data.DataLoader(
        ExpertDataset(
        file_name, num_trajectories=2, subsample_frequency=1),
        batch_size=128,
        shuffle=True,
        drop_last=True)

    learn_bc(actor_critic, device, gail_train_loader, gail_val_loader)