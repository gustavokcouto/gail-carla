"""The main framwork for this work.
See README for usage.
"""

import argparse
import torch
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# GAIL baseline
# python main.py --env-name CarRacing-v0 --algo ppo --gail --gail-experts-dir /serverdata/rohit/BCGAIL/CarRacingPPO/ --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --model_name CarRacingGAIL --gail-batch-size 32
# python main.py --env-name CarRacing-v0 --algo ppo --gail --gail-experts-dir /serverdata/rohit/BCGAIL/CarRacingPPO/ --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --model_name CarRacingBCGAIL0.125 --gail-batch-size 32 --bcgail 1 --gailgamma 0.125 --decay 1 --num-env-steps 1000000 --seed 1

# set key parameters


def read_params():
    params = {
        # environment ID
        'env_name': 'carla',
        # algorithm ID
        'algo': 'WDAIL',

        # general
        # total steps
        'num_env_steps': 10e6,
        # num_model
        'cuda': 0,
        # seed
        'seed': 1,
        # use linear lr decay
        'use_linear_lr_decay': False,

        # ppo
        # num_processes
        'num_processes': 10,
        # num-steps
        'num_steps': 2400,
        # learning rate
        'lr': 2.5e-4,
        # ppo epoch num
        'ppo_epoch': 4,
        # number of batches for ppo (default: 32)
        'num_mini_batch': 8,
        # ppo clip parameter (default: 0.2)
        'clip_param': 0.1,
        # ADAM optimizer epsilon (default: 1e-5)
        'eps': 1e-5,
        # ADAM Optimizer betas param
        'betas': [0.9, 0.99],
        # discount factor for rewards (default: 0.99)
        'gamma': 0.99,
        # gae lambda parameter (default: 0.95)
        'gae_lambda': 0.95,
        # entropy term coefficient (default: 0.01)
        'entropy_coef': 0.0,
        # variable entropy (std dev as net parameter)
        'var_ent': False,
        # value loss coefficient (default: 0.5)
        'value_loss_coef': 0.5,
        # max norm of gradients (default: 0.5)
        'max_grad_norm': 0.5,
        # Model log std deviation
        'std_dev': [
            {
                'logstd': [-0.6, -0.2],
                'limit': 100
            },
            {
                'logstd': [-1.4, -1.0],
                'limit': 200
            },
            {
                'logstd': [-2.0, -1.8]
            }
        ],

        # gail
        # directory that contains expert demonstrations for gail
        'gail_experts_dir': './gail_experts',
        # gail batch size (default: 128)
        'gail_batch_size': 128,
        # gail learning rate
        'gail_lr': 2.5e-4,
        # ADAM optimizer epsilon (default: 1e-5)
        'gail_eps': 1e-8,
        # GAIL Optimizer betas param
        'gail_betas': [0.9, 0.99],
        # duration of gail pre epoch
        'gail_thre': 5,
        # number of steps to train discriminator during pre epoch
        'gail_pre_epoch': 25,
        # number of steps to train discriminator in each epoch
        'gail_epoch': 5,
        # max norm of gradients (default: 0.5)
        'gail_max_grad_norm': 0.5,
        # num trajs
        'num_trajs': 3,
        # trajectories subsample frequency
        'subsample_frequency': 1,
        # log interval, one log per n updates (default: 10)
        'log_interval': 1,
        # eval interval, one eval per n updates (default: 10)
        'eval_interval': 3,

        # bcgail
        'bcgail': 0,
        'decay': 0.99,
        'gailgamma': 0.125,
        # Use final activation? (Useful for certain scenarios)
        'use_activation': True
    }

    config_file = open('params.json')
    config = json.load(config_file)
    params.update(config)
    return params


def train(params):

    # from ppo_gail_iko.algo.ppo4multienvs import PPO, ReplayBuffer
    from algo.ppo import PPO
    from tools.model import Policy

    from algo.wdgail import Discriminator, ExpertDataset

    from tools.learn import gailLearning_mujoco_origin
    from learn_bc import learn_bc

    from tools import utli
    from tools import utils
    from tools.envs import make_vec_envs

    from carla_env import CarlaEnv

    from collections import deque
    import time
    import numpy as np

    # from nets.network import ActorCritic_mujoco as ActorCritic
    run_params = params
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    torch.cuda.manual_seed_all(params['seed'])

    device = torch.device(
        'cuda:' + str(params['cuda']) if torch.cuda.is_available() else 'cpu')

    file_name = os.path.join(
        params['gail_experts_dir'], "trajs_{}.pt".format(
            params['env_name'].split('-')[0].lower()))

    gail_train_loader = torch.utils.data.DataLoader(
        ExpertDataset(
            file_name, num_trajectories=params['num_trajs'], subsample_frequency=params['subsample_frequency']),
        batch_size=params['gail_batch_size'],
        shuffle=True,
        drop_last=True)

    envs = make_vec_envs(params['num_processes'], device)
    env_eval = CarlaEnv(eval=True)

    # network
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.metrics_space,
        envs.action_space,
        params['use_activation'],
        params['std_dev'],
        params['var_ent'])
    actor_critic.to(device)

    # learn_bc(actor_critic, envs, device, gail_train_loader)

    agent = PPO(
        actor_critic,
        params['clip_param'],
        params['ppo_epoch'],
        params['num_mini_batch'],
        params['value_loss_coef'],
        params['entropy_coef'],
        device,
        lr=params['lr'],
        eps=params['eps'],
        betas=params['betas'],
        gamma=params['gailgamma'],
        decay=params['decay'],
        act_space=envs.action_space,
        max_grad_norm=params['max_grad_norm'])

    # discriminator
    discr = Discriminator(
        envs.observation_space.shape,
        envs.metrics_space,
        envs.action_space,
        100,
        device,
        params['gail_lr'],
        params['gail_eps'],
        params['gail_betas'],
        params['gail_max_grad_norm'])

    model = gailLearning_mujoco_origin(run_params=run_params,
                                       envs=envs,
                                       env_eval=env_eval,
                                       actor_critic=actor_critic,
                                       agent=agent,
                                       discriminator=discr,
                                       gail_train_loader=gail_train_loader,
                                       device=device,
                                       utli=utli)

    return 0


def main(params):

    model, env = train(params)

    return model


if __name__ == '__main__':

    params = read_params()
    main(params)
