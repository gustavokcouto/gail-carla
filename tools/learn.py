import os
import shutil
from tensorboardX import SummaryWriter
import torch
import numpy as np
from collections import deque
import time
from tools import utils, utli
from carla_env import CarlaEnv


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def gailLearning_mujoco_origin(cl_args, env, actor_critic, agent, discriminator, rollouts, gail_train_loader, device, utli):

    log_save_name = utli.Log_save_name4gail(cl_args)
    log_save_path = os.path.join("./runs", log_save_name)
    if os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    utli.writer = SummaryWriter(log_save_path)

    # Evaluate the initial network
    evaluations = []
    # begin optimize

    reward_window4Evaluate = deque(maxlen=10)
    time_step = 0
    episode_t = 0
    episode_timesteps = 0
    count = 0

    # begin optimize
    nsteps = cl_args.num_steps
    S_time = time.time()

    nenv = 1

    nbatch = np.floor(nsteps/nenv)
    nbatch = nbatch.astype(np.int16)
    nupdates = np.floor(cl_args.num_env_steps / nsteps)
    nupdates = nupdates.astype(np.int16)

    epinfobuf = deque(maxlen=10)

    epgailbuf = deque(maxlen=10)

    episode_rewards = deque(maxlen=10)

    cum_gailrewards = .0

    i_update = 0

    obs, metrics = env.reset()
    obs = torch.from_numpy(obs).float().to(device)
    obs = torch.stack([obs])
    metrics = torch.from_numpy(metrics).float().to(device)
    metrics = torch.stack([metrics])
    rollouts.obs[0].copy_(obs)
    rollouts.metrics[0].copy_(metrics)
    rollouts.to(device)

    start = time.time()

    while i_update < nupdates:

        episode_t += 1
        i_update += 1
        epinfos = []

        if cl_args.use_linear_lr_decay:
            # decrease learning rate linearly
            utli.update_linear_schedule(
                agent.optimizer, i_update, nupdates,
                cl_args.lr)

        for step in range(nbatch):
            time_step += 1

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.obs[step], rollouts.metrics[step], rollouts.masks[step])

            obs, metrics, reward, done, info = env.step(action)
            obs = torch.from_numpy(obs).float().to(device)
            obs = torch.stack([obs])
            metrics = torch.from_numpy(metrics).float().to(device)
            metrics = torch.stack([metrics])
            reward = torch.from_numpy(reward).float()

            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            if done:
                mask = torch.FloatTensor([0.0])
                obs, metrics = env.reset()
                obs = torch.from_numpy(obs).float()
                obs = torch.stack([obs])
                metrics = torch.from_numpy(metrics).float()
                metrics = torch.stack([metrics])
            else:
                mask = torch.FloatTensor([1.0])
            rollouts.insert(obs, metrics, action, action_log_prob, value, reward, mask)
        print('finished sim')
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.metrics[-1], rollouts.masks[-1]).detach()

        # gail
        if cl_args.gail:
            gail_epoch = cl_args.gail_epoch

            dis_total_losses = []
            policy_rewards = []
            expert_rewards = []
            dis_losses = []
            dis_gps = []
            expert_losses = []
            policy_losses = []
            for _ in range(gail_epoch):

                dis_total_loss, policy_mean_reward, expert_reward_mean, dis_loss, dis_gp, expert_loss, policy_loss = discriminator.update(gail_train_loader, rollouts)
                dis_total_losses.append(dis_total_loss)
                policy_rewards.append(policy_mean_reward)
                expert_rewards.append(expert_reward_mean)
                dis_losses.append(dis_loss)
                dis_gps.append(dis_gp)
                expert_losses.append(expert_loss)
                policy_losses.append(policy_loss)

            utli.recordDisLossResults(results=(np.mean(np.array(dis_total_losses)),
                                               np.mean(np.array(policy_rewards)),
                                               np.mean(np.array(expert_rewards)),
                                               np.mean(np.array(dis_losses)),
                                               np.mean(np.array(dis_gps)),
                                               np.mean(np.array(expert_losses)),
                                               np.mean(np.array(policy_losses))),
                                      time_step=time_step)


            for step in range(nbatch):
                rollouts.rewards[step] = discriminator.predict_reward(
                    rollouts.obs[step], rollouts.metrics[step], rollouts.actions[step], cl_args.gamma,
                    rollouts.masks[step])
                if rollouts.masks[step] == 1:
                    cum_gailrewards += rollouts.rewards[step].item()
                else:
                    epgailbuf.append(cum_gailrewards)
                    cum_gailrewards=.0

        # compute returns
        rollouts.compute_returns(next_value, cl_args.gamma, cl_args.gae_lambda)

        # training PPO policy

        value_loss, action_loss, dist_entropy, bc_loss, gail_loss, gail_gamma = agent.update(rollouts)

        utli.recordLossResults(results=(value_loss,
                                        action_loss,
                                        dist_entropy,
                                        bc_loss,
                                        gail_loss,
                                        gail_gamma),
                               time_step=time_step)
        rollouts.after_update()


        epinfobuf.extend(epinfos)
        if not len(epinfobuf):
            continue
        eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
        eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])

        utli.recordTrainResults_gail(results=(eprewmean,
                                              eplenmean,
                                              np.mean(np.array(epgailbuf))),
                                time_step=time_step)


        print("Episode: %d,   Time steps: %d,   Mean length: %d    Mean Reward: %f    Mean Gail Reward:%f"
            % (episode_t, time_step, eplenmean, eprewmean, np.mean(np.array(epgailbuf))))

        if i_update % cl_args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (i_update + 1) * cl_args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(i_update, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

    return 0
