import numpy as np
import time
from pathlib import Path
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers.monitoring.video_recorder import ImageEncoder


class WandbCallback(BaseCallback):
    def __init__(self, vec_env):
        super(WandbCallback, self).__init__(verbose=1)

        # self._video_path = Path('video')
        # self._video_path.mkdir(parents=True, exist_ok=True)
        # self._ckpt_dir = Path('ckpt')
        # self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        wandb.init(project='gail_carla', name=None, notes=None, tags=None)

        # wandb.save('./config_agent.yaml')
        # wandb.save('.hydra/*')

        self.vec_env = vec_env

        self._eval_step = int(1e5)
        self._buffer_step = int(1e5)

    def _init_callback(self):
        self.n_epoch = 0
        self._last_time_buffer = self.model.num_timesteps
        self._last_time_eval = self.model.num_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self):
        # self.model._last_obs = self.model.env.reset()
        pass

    def _on_training_end(self) -> None:
        print(f'n_epoch: {self.n_epoch}, num_timesteps: {self.model.num_timesteps}')
        # save time
        time_elapsed = time.time() - self.model.start_time
        wandb.log({
            'time/n_epoch': self.n_epoch,
            'time/sec_per_epoch': time_elapsed / (self.n_epoch+1),
            'time/fps': (self.model.num_timesteps-self.model.start_num_timesteps) / time_elapsed,
            'time/train': self.model.t_train,
            'time/train_values': self.model.t_train_values,
            'time/rollout': self.model.t_rollout
        }, step=self.model.num_timesteps)
        wandb.log(self.model.train_debug, step=self.model.num_timesteps)

        # evaluate and save checkpoint
        # if (self.model.num_timesteps - self._last_time_eval) >= self._eval_step:
        #     self._last_time_eval = self.model.num_timesteps
        #     # evaluate
        #     eval_video_path = (self._video_path / f'eval_{self.model.num_timesteps}.mp4').as_posix()
        #     avg_ep_stat, ep_events = self.evaluate_policy(self.vec_env, self.model.policy, eval_video_path)
        #     # log to wandb
        #     wandb.log({f'video/{self.model.num_timesteps}': wandb.Video(eval_video_path)},
        #               step=self.model.num_timesteps)
        #     wandb.log(avg_ep_stat, step=self.model.num_timesteps)

        #     ckpt_path = (self._ckpt_dir / f'ckpt_{self.model.num_timesteps}.pth').as_posix()
        #     self.model.save(ckpt_path)
        #     wandb.save(f'./{ckpt_path}')
        self.n_epoch += 1

    def _on_rollout_end(self):
        wandb.log({'time/rollout': self.model.t_rollout}, step=self.model.num_timesteps)

        # save rollout statistics
        avg_ep_stat = self.get_avg_ep_stat(self.model.ep_stat_buffer, prefix='rollout/')
        wandb.log(avg_ep_stat, step=self.model.num_timesteps)

        # action, mu, sigma histogram
        action_statistics = np.array(self.model.action_statistics)
        mu_statistics = np.array(self.model.mu_statistics)
        sigma_statistics = np.array(self.model.sigma_statistics)
        n_action = action_statistics.shape[-1]
        action_statistics = action_statistics.reshape(-1, n_action)
        mu_statistics = mu_statistics.reshape(-1, n_action)
        sigma_statistics = sigma_statistics.reshape(-1, n_action)

        for i in range(n_action):
            wandb.log({f'action[{i}]': wandb.Histogram(action_statistics[:, i])}, step=self.model.num_timesteps)
            wandb.log({f'alpha[{i}]': wandb.Histogram(mu_statistics[:, i])}, step=self.model.num_timesteps)
            wandb.log({f'beta[{i}]': wandb.Histogram(sigma_statistics[:, i])}, step=self.model.num_timesteps)

        # render buffer
        # if (self.model.num_timesteps - self._last_time_buffer) >= self._buffer_step:
        #     self._last_time_buffer = self.model.num_timesteps
        #     buffer_video_path = (self._video_path / f'buffer_{self.model.num_timesteps}.mp4').as_posix()

        #     list_buffer_im = self.model.buffer.render()
        #     encoder = ImageEncoder(buffer_video_path, list_buffer_im[0].shape, 30, 30)
        #     for im in list_buffer_im:
        #         encoder.capture_frame(im)
        #     encoder.close()
        #     encoder = None

        #     wandb.log({f'buffer/{self.model.num_timesteps}': wandb.Video(buffer_video_path)},
        #               step=self.model.num_timesteps)

    @staticmethod
    def evaluate_policy(env, policy, video_path, min_eval_steps=3000):
        policy = policy.eval()
        t0 = time.time()
        for i in range(env.num_envs):
            env.set_attr('eval_mode', True, indices=i)
        obs = env.reset()

        list_render = []
        ep_stat_buffer = []
        ep_events = {}
        for i in range(env.num_envs):
            ep_events[f'venv_{i}'] = []

        n_step = 0
        n_timeout = 0
        env_done = np.array([False]*env.num_envs)
        # while n_step < min_eval_steps:
        while n_step < min_eval_steps or not np.all(env_done):
            actions, values, log_probs, mu, sigma, _ = policy.forward(obs, deterministic=True, clip_action=True)
            obs, reward, done, info = env.step(actions)

            for i in range(env.num_envs):
                env.set_attr('action_value', values[i], indices=i)
                env.set_attr('action_log_probs', log_probs[i], indices=i)
                env.set_attr('action_mu', mu[i], indices=i)
                env.set_attr('action_sigma', sigma[i], indices=i)

            list_render.append(env.render(mode='rgb_array'))

            n_step += 1
            env_done |= done

            for i in np.where(done)[0]:
                ep_stat_buffer.append(info[i]['episode_stat'])
                ep_events[f'venv_{i}'].append(info[i]['episode_event'])
                n_timeout += int(info[i]['timeout'])

        # conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
        encoder = ImageEncoder(video_path, list_render[0].shape, 30, 30)
        for im in list_render:
            encoder.capture_frame(im)
        encoder.close()

        avg_ep_stat = WandbCallback.get_avg_ep_stat(ep_stat_buffer, prefix='eval/')
        avg_ep_stat['eval/eval_timeout'] = n_timeout

        duration = time.time() - t0
        avg_ep_stat['time/t_eval'] = duration
        avg_ep_stat['time/fps_eval'] = n_step * env.num_envs / duration

        for i in range(env.num_envs):
            env.set_attr('eval_mode', False, indices=i)
        obs = env.reset()
        return avg_ep_stat, ep_events

    @staticmethod
    def get_avg_ep_stat(ep_stat_buffer, prefix=''):
        avg_ep_stat = {}
        if len(ep_stat_buffer) > 0:
            for ep_info in ep_stat_buffer:
                for k, v in ep_info.items():
                    k_avg = f'{prefix}{k}'
                    if k_avg in avg_ep_stat:
                        avg_ep_stat[k_avg] += v
                    else:
                        avg_ep_stat[k_avg] = v

            n_episodes = float(len(ep_stat_buffer))
            for k in avg_ep_stat.keys():
                avg_ep_stat[k] /= n_episodes
            avg_ep_stat[f'{prefix}n_episodes'] = n_episodes

        return avg_ep_stat