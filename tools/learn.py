import os
import shutil
from tensorboardX import SummaryWriter
import torch
import numpy as np
from collections import deque
import time
from tools import utils, utli
from tools.storage import RolloutStorage


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def gailLearning_mujoco_origin(cl_args, envs, actor_critic, agent, discriminator, gail_train_loader, device, utli):

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

    nenv = cl_args.num_processes

    nbatch = np.floor(nsteps/nenv)
    nbatch = nbatch.astype(np.int16)

    # The buffer
    rollouts = RolloutStorage(nbatch,
                              cl_args.num_processes,
                              envs.observation_space.shape,
                              envs.metrics_space.shape,
                              envs.action_space)

    nupdates = np.floor(cl_args.num_env_steps / nsteps)
    nupdates = nupdates.astype(np.int16)

    epinfobuf = deque(maxlen=10)

    epgailbuf = deque(maxlen=10)

    episode_rewards = deque(maxlen=10)

    cum_gailrewards = [.0 for _ in range(cl_args.num_processes)]

    i_update = 0

    obs, metrics = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.metrics[0].copy_(metrics)

    start = time.time()

    best_episode = 0
    while i_update < nupdates:

        episode_t += 1
        i_update += 1
        epinfos = []

        if cl_args.use_linear_lr_decay:
            # decrease learning rate linearly
            utli.update_linear_schedule(
                agent.optimizer, i_update, nupdates,
                cl_args.lr)

        actor_critic.to(device)
        for step in range(nbatch):
            time_step += 1

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(rollouts.obs[step].to(device), rollouts.metrics[step].to(device))

            obs, metrics, rewards, done, infos = envs.step(action)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            rollouts.insert(obs, metrics, action.cpu(), action_log_prob.cpu(), value.cpu(), rewards, masks)

        print('finished sim')
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1].to(device), rollouts.metrics[-1].to(device)).detach()
        actor_critic.cpu()

        # gail
        discriminator.gpu()
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
                                    time_step=i_update)


        for step in range(nbatch):
            rollouts.rewards[step] = discriminator.predict_reward(
                rollouts.obs[step].to(device),
                rollouts.metrics[step].to(device),
                rollouts.actions[step].to(device),
                cl_args.gamma,
                rollouts.masks[step].to(device))

            for i_env in range(cl_args.num_processes):
                if rollouts.masks[step][i_env] == 0:
                        cum_gailrewards[i_env] += rollouts.rewards[step][i_env].item()
                else:
                    epgailbuf.append(cum_gailrewards[i_env])
                    cum_gailrewards[i_env]=.0
        discriminator.cpu()

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
                               time_step=i_update)
        rollouts.after_update()


        epinfobuf.extend(epinfos)
        if not len(epinfobuf):
            continue
        eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
        eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])

        utli.recordTrainResults_gail(results=(eprewmean,
                                              eplenmean,
                                              np.mean(np.array(epgailbuf))),
                                time_step=i_update)

        if eplenmean > best_episode:
            best_episode = eplenmean
            torch.save(actor_critic.state_dict(), 'carla_actor.pt')

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
