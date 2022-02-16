import gym
from carla_gym.utils import config_utils
from carla_gym.core.task_actor.scenario_actor.agents.basic_agent import BasicAgent
import carla
from PIL import Image


obs_configs = {
    'hero': {
        'speed': {
            'module': 'actor_state.speed'
        },
        'gnss': {
            'module': 'navigation.gnss'
        },
        'central_rgb': {
            'module': 'camera.rgb',
            'fov': 60,
            'width': 384,
            'height': 216,
            'location': [0.8, 0.0, 1.3],
            'rotation': [0.0, 0.0, 0.0]
        },
        'left_rgb': {
            'module': 'camera.rgb',
            'fov': 60,
            'width': 384,
            'height': 216,
            'location': [0.8, 0.0, 1.3],
            'rotation': [0.0, 0.0, -55.0]
        },
        'right_rgb': {
            'module': 'camera.rgb',
            'fov': 60,
            'width': 384,
            'height': 216,
            'location': [0.8, 0.0, 1.3],
            'rotation': [0.0, 0.0, 55.0]
        },
        'route_plan': {
            'module': 'navigation.waypoint_plan',
            'steps': 20
        }
    }
}
reward_configs = {
    'hero': {
        'entry_point': 'reward.valeo_action:ValeoAction'
    }
}
terminal_configs = {
    'hero': {
        'entry_point': 'terminal.leaderboard_dagger:LeaderboardDagger',
        'kwargs': {
            'max_time': 900,
            'no_collision': False,
            'no_run_rl': False,
            'no_run_stop': False,
        }
    }
}
env_configs = {
    'carla_map': 'Town01',
    'num_zombie_vehicles': 0,
    'num_zombie_walkers': 0,
    'weather_group': 'train'
}
env = gym.make('Endless-v0', obs_configs=obs_configs, reward_configs=reward_configs,
                   terminal_configs=terminal_configs, host='localhost', port=2000,
                   seed=2021, no_rendering=False, **env_configs)

obs = env.reset()
for _ in range(10):
    basic_agent = BasicAgent(env._ev_handler.ego_vehicles['hero'], None, 6.0)
    action = basic_agent.get_action()
    vehicle_control = carla.VehicleControl(steer=action[0], throttle=action[1], brake=action[2])
    driver_control = {'hero': vehicle_control}
    new_obs, reward, done, info = env.step(driver_control)
    rgb = new_obs['hero']['central_rgb']['data']
    Image.fromarray(rgb).save('rgb.png')