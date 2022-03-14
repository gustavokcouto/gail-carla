import carla
from carla_gym.envs import EndlessEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from rl_birdview.utils.rl_birdview_wrapper import RlBirdviewWrapper
from rl_birdview.models.ppo import PPO
from rl_birdview.models.ppo_policy import PpoPolicy
from rl_birdview.utils.wandb_callback import WandbCallback
from stable_baselines3.common.callbacks import CallbackList


obs_configs = {
    'hero': {
        'speed': {
            'module': 'actor_state.speed'
        },
        'control': {
            'module': 'actor_state.control'
        },
        'velocity': {
            'module': 'actor_state.velocity'
        },
        'birdview': {
            'module': 'birdview.chauffeurnet',
            'width_in_pixels': 192,
            'pixels_ev_to_bottom': 40,
            'pixels_per_meter': 5.0,
            'history_idx': [-16, -11, -6, -1],
            'scale_bbox': True,
            'scale_mask_col': 1.0
        },
        'route_plan': {
            'module': 'navigation.waypoint_plan',
            'steps': 20
        }
    }
}

reward_configs = {
    'hero': {
        'entry_point': 'reward.valeo_action:ValeoAction',
        'kwargs': {}
    }
}

terminal_configs = {
    'hero': {
        'entry_point': 'terminal.valeo_no_det_px:ValeoNoDetPx',
        'kwargs': {}
    }
}

env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': [0, 150],
    'num_zombie_walkers': [0, 300],
    'weather_group': 'dynamic_1.0'
}

env_wrapper_configs = {
    'input_states': ['control', 'vel_xy'],
    'acc_as_action': True
}

multi_env_configs = [
        {"host": "192.168.0.4", "port": 2000},
        {"host": "192.168.0.4", "port": 2002},
        {"host": "192.168.0.4", "port": 2004},
        {"host": "192.168.0.4", "port": 2006},
        {"host": "192.168.0.4", "port": 2008},
        {"host": "192.168.0.5", "port": 2012},
        {"host": "192.168.0.5", "port": 2014},
        {"host": "192.168.0.5", "port": 2016},
        {"host": "192.168.0.5", "port": 2018},
        {"host": "192.168.0.5", "port": 2020}
]

def env_maker(config):
    env = EndlessEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                    terminal_configs=terminal_configs, host=config['host'], port=config['port'],
                    seed=2021, no_rendering=True, **env_configs)
    env = RlBirdviewWrapper(env, **env_wrapper_configs)
    return env


if __name__ == '__main__':
    env = SubprocVecEnv([lambda config=config: env_maker(config) for config in multi_env_configs])
    policy = PpoPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy_head_arch=[256, 256],
        value_head_arch=[256, 256],
        features_extractor_entry_point='rl_birdview.models.torch_layers:XtMaCNN',
        features_extractor_kwargs={'states_neurons': [256,256]},
        distribution_entry_point='rl_birdview.models.distributions:BetaDistribution'
    )
    agent = PPO(
        policy=policy,
        env=env,
        learning_rate=1e-5,
        n_steps_total=12288,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.9,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        explore_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        update_adv=False,
        lr_schedule_step=8
    )
    wb_callback = WandbCallback(env)
    callback = CallbackList([wb_callback])
    agent.learn(total_timesteps=1e8, callback=callback)
