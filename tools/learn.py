import os
import shutil
from pathlib import Path

from tensorboardX import SummaryWriter
import torch
import numpy as np
from collections import deque
import time
from tools import utils, utli
from tools.storage import RolloutStorage
from tools.envs import EnvEpoch


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def gailLearning_mujoco_origin(run_params,
                               envs,
                               env_eval,
                               actor_critic,
                               agent,
                               discriminator,
                               gail_train_loader,
                               device,
                               utli
                               ):

    log_save_name = utli.Log_save_name4gail(run_params)
    log_save_path = os.path.join("./runs", log_save_name)
    if not run_params['resume_training'] and os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    utli.writer = SummaryWriter(log_save_path)

    # begin optimize

    time_step = 0

    # begin optimize
    nsteps = run_params['num_steps']

    nenv = len(run_params['envs_params'])

    nbatch = np.floor(nsteps/nenv)
    nbatch = nbatch.astype(np.int16)

    obs_shape = envs.observation_space.shape
    metrics_shape = envs.metrics_space.shape
    action_shape = envs.action_space.shape
    # The buffer
    rollouts = RolloutStorage(nbatch,
                              len(run_params['envs_params']),
                              obs_shape,
                              metrics_shape,
                              action_shape)

    rollout_eval = RolloutStorage(env_eval.ep_length,
                                  1,
                                  obs_shape,
                                  metrics_shape,
                                  action_shape)

    nupdates = np.floor(run_params['num_env_steps'] / nsteps)
    nupdates = nupdates.astype(np.int16)

    cum_gailrewards = [.0 for _ in range(len(run_params['envs_params']))]

    i_update = 0

    obs, metrics = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.metrics[0].copy_(metrics)

    start = time.time()

    steps_eval = 0
    eval_reward = None

    model_path = Path('gail_model.pt')
    if run_params['resume_training']:
        load_data = torch.load(model_path)
        actor_critic.load_state_dict(load_data[0])
        discriminator.load_state_dict(load_data[1])
        i_update = load_data[2]
        start -= load_data[3]

    while i_update < nupdates:
        epinfobuf = []

        epgailbuf = []

        episode_rewards = []
        routes_rewards = {}
        for route_idx in run_params['routes']:
            routes_rewards[route_idx] = []

        i_update += 1
        epinfos = []

        if run_params['use_linear_lr_decay']:
            # decrease learning rate linearly
            utli.update_linear_schedule(
                agent.optimizer, i_update, nupdates,
                run_params['lr'])

        discriminator.cpu()
        actor_critic.to(device)
        EnvEpoch.set_epoch(i_update)
        for step in range(nbatch):
            time_step += 1

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.obs[step].to(device), rollouts.metrics[step].to(device))

            obs, metrics, rewards, done, infos = envs.step(action)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
                    routes_rewards[info['route_id']].append(info['episode']['r'])
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            rollouts.insert(obs.cpu(), metrics.cpu(), action.cpu(),
                            action_log_prob, value, rewards, masks)

        print('finished sim')

        with torch.no_grad():
            rollouts.value_preds[-1] = actor_critic.get_value(
                rollouts.obs[-1].to(device), rollouts.metrics[-1].to(device)).detach()
        actor_critic.cpu()
        discriminator.to(device)

        # gail
        disc_pre_loss, expert_pre_reward, policy_pre_reward = discriminator.compute_loss(
            gail_train_loader, rollouts)
        gail_epoch = run_params['gail_epoch']
        if i_update < run_params['gail_thre']:
            gail_epoch += (run_params['gail_pre_epoch'] - run_params['gail_epoch']) * \
                (run_params['gail_thre'] - (i_update - 1)) / \
                run_params['gail_thre']  # Warm up
            gail_epoch = int(gail_epoch)
        dis_total_losses = []
        policy_rewards = []
        expert_rewards = []
        dis_losses = []
        dis_gps = []
        expert_losses = []
        policy_losses = []
        for _ in range(gail_epoch):
            (
                dis_total_loss,
                policy_mean_reward,
                expert_reward_mean,
                dis_loss,
                dis_gp,
                expert_loss,
                policy_loss
            ) = discriminator.update(
                gail_train_loader, rollouts)
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
                                           np.mean(np.array(policy_losses)),
                                           disc_pre_loss,
                                           expert_pre_reward,
                                           policy_pre_reward),
                                  time_step=i_update)
        for step in range(nbatch):
            rollouts.gail_rewards[step] = discriminator.predict_reward(
                rollouts.obs[step],
                rollouts.metrics[step],
                rollouts.actions[step],
                run_params['gamma'],
                rollouts.masks[step])

            for i_env in range(len(run_params['envs_params'])):
                if rollouts.masks[step][i_env]:
                    cum_gailrewards[i_env] += rollouts.gail_rewards[step][i_env].item()
                else:
                    epgailbuf.append(cum_gailrewards[i_env])
                    cum_gailrewards[i_env] = .0

        # compute returns
        rollouts.compute_returns(run_params['gamma'], run_params['gae_lambda'])

        discriminator.cpu()
        actor_critic.to(device)

        # training PPO policy
        if run_params['bcgail']:
            value_loss, action_loss, dist_entropy, bc_loss, gail_loss, gail_gamma, steer_std, throttle_std = agent.update(
                rollouts, gail_train_loader)
        else:
            value_loss, action_loss, dist_entropy, bc_loss, gail_loss, gail_gamma, steer_std, throttle_std = agent.update(
                rollouts)

        if i_update % run_params['eval_interval'] == 0 or eval_reward is None:
            done = False
            obs, metrics = env_eval.reset()
            steps_eval = 0
            while not done:
                obs = obs.to(device)
                obs = torch.stack([obs])
                metrics = metrics.to(device)
                metrics = torch.stack([metrics])
                with torch.no_grad():
                    value, actions, action_log_prob = actor_critic.act(
                        obs,
                        metrics,
                        deterministic=True
                    )
                rollout_eval.obs[steps_eval].copy_(obs.cpu())
                rollout_eval.metrics[steps_eval].copy_(metrics.cpu())
                rollout_eval.actions[steps_eval].copy_(actions.cpu())

                action = actions.cpu().numpy()[0]
                obs, metrics, _, done, info = env_eval.step(action)
                steps_eval += 1
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    eval_reward = info['episode']['r']

            obs = torch.stack([obs])
            metrics = torch.stack([metrics])
            rollout_eval.obs[steps_eval].copy_(obs.cpu())
            rollout_eval.metrics[steps_eval].copy_(metrics.cpu())
            actor_critic.cpu()
            discriminator.to(device)
            disc_eval_loss, expert_eval_reward, policy_eval_reward = discriminator.compute_loss(
                gail_train_loader, rollout_eval, batch_size=steps_eval-1)

        utli.recordLossResults(results=(value_loss,
                                        action_loss,
                                        dist_entropy,
                                        bc_loss,
                                        gail_loss,
                                        gail_gamma,
                                        steer_std,
                                        throttle_std),
                               time_step=i_update)
        rollouts.after_update()

        epinfobuf.extend(epinfos)
        if not len(epinfobuf):
            continue
        eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
        eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])

        utli.recordTrainResults_gail(results=(eprewmean,
                                              eplenmean,
                                              np.mean(np.array(epgailbuf)),
                                              steps_eval,
                                              eval_reward,
                                              disc_eval_loss,
                                              expert_eval_reward,
                                              policy_eval_reward
                                              ),
                                     time_step=i_update)

        utli.record_routes_rewards(routes_rewards, i_update)

        time_diff = time.time() - start
        torch.save([actor_critic.state_dict(), discriminator.state_dict(), i_update, time_diff], model_path)

        print("Episode: %d,   Time steps: %d,   Mean length: %d    Mean Reward: %f    Mean Gail Reward:%f"
              % (i_update, time_step, eplenmean, eprewmean, np.mean(np.array(epgailbuf))))

        if i_update % run_params['log_interval'] == 0 and len(episode_rewards) > 1:
            total_num_steps = (i_update + 1) * run_params['num_steps']
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