"""The main framwork for this work.
See README for usage.
"""

import argparse
import torch

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## GAIL baseline
# python main.py --env-name CarRacing-v0 --algo ppo --gail --gail-experts-dir /serverdata/rohit/BCGAIL/CarRacingPPO/ --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --model_name CarRacingGAIL --gail-batch-size 32
# python main.py --env-name CarRacing-v0 --algo ppo --gail --gail-experts-dir /serverdata/rohit/BCGAIL/CarRacingPPO/ --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --model_name CarRacingBCGAIL0.125 --gail-batch-size 32 --bcgail 1 --gailgamma 0.125 --decay 1 --num-env-steps 1000000 --seed 1

# set key parameters
def argsparser():
    parser = argparse.ArgumentParser("WDAIL")
    parser.add_argument('--env_name', help='environment ID', default='carla')
    parser.add_argument('--algo', help='algorithm ID', default='WDAIL')
    # general
    parser.add_argument('--num_env_steps', help='total steps', type=int, default=108e4)
    parser.add_argument('--cuda', help='num_model', type=int, default=0)
    parser.add_argument('--seed', help='seed', type=int, default=1)
    parser.add_argument('--use_linear_lr_decay', help='use linear lr decay', type=bool, default=False)

    #ppo
    parser.add_argument('--num_processes', help='num_processes', type=int, default=8)
    parser.add_argument('--num-steps', help='num-steps', type=int, default=3600)
    parser.add_argument('--lr', help='learning rate', type=float, default=2.5e-4)
    parser.add_argument('--init_entropy', help='initial entropy', type=float, default=2.5)
    parser.add_argument('--end_entropy', help='end entropy', type=float, default=-2)
    parser.add_argument('--ppo_epoch', help='ppo epoch num', type=int, default=4)
    parser.add_argument('--num-mini-batch', type=int, default=8, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.1, help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.00, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')

    # gail
    parser.add_argument('--expert_path', help='trajs path', type=str, default='../data/ikostirkov/trajs_ant.h5')
    parser.add_argument('--gail-experts-dir',default='./gail_experts', help='directory that contains expert demonstrations for gail')
    parser.add_argument('--gail_batch_size', type=int, default=128, help='gail batch size (default: 128)')
    parser.add_argument('--gail_epoch', help='number of steps to train discriminator in each epoch', type=int, default=5)
    parser.add_argument('--gail-max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num_trajs', help='num trajs', type=int, default=3)
    parser.add_argument('--subsample_frequency', help='num trajs', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=1, help='log interval, one log per n updates (default: 10)')

    parser.add_argument('--bcgail', type=int, default=0)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--gailgamma', type=float, default=0.125)
    parser.add_argument('--use_activation', default=0, type=int, help='Use final activation? (Useful for certain scenarios)')
    return parser.parse_args()

def train(args):

    # from ppo_gail_iko.algo.ppo4multienvs import PPO, ReplayBuffer
    from algo.ppo import PPO
    from tools.model import Policy

    from algo.wdgail import Discriminator, ExpertDataset

    from tools.learn import gailLearning_mujoco_origin
    from learn_bc import learn_bc

    from tools import utli
    from tools import utils
    from tools.envs import make_vec_envs

    from collections import deque
    import time
    import numpy as np


    # from nets.network import ActorCritic_mujoco as ActorCritic
    cl_args = args
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda:'+ str(cl_args.cuda) if torch.cuda.is_available() else 'cpu')

    file_name = os.path.join(
        args.gail_experts_dir, "trajs_{}.pt".format(
            args.env_name.split('-')[0].lower()))

    gail_train_loader = torch.utils.data.DataLoader(
        ExpertDataset(
        file_name, num_trajectories=args.num_trajs, subsample_frequency=args.subsample_frequency),
        batch_size=args.gail_batch_size,
        shuffle=True,
        drop_last=True)

    envs = make_vec_envs(args.num_processes, device)

    activation = None
    if args.use_activation:
        activation = torch.tanh

    # network
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.metrics_space,
        envs.action_space,
        activation=activation)
    actor_critic.to(device)

    # learn_bc(actor_critic, envs, device, gail_train_loader)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        device,
        lr=args.lr,
        eps=args.eps,
        gamma=args.gailgamma,
        decay=args.decay,
        act_space=envs.action_space,
        max_grad_norm=args.max_grad_norm)

    # discriminator
    discr = Discriminator(
        envs.observation_space.shape,
        envs.metrics_space,
        envs.action_space,
        100,
        device,
        args.gail_max_grad_norm)

    model = gailLearning_mujoco_origin(cl_args=cl_args,
                                       envs=envs,
                                       actor_critic=actor_critic,
                                       agent=agent,
                                       discriminator=discr,
                                       gail_train_loader=gail_train_loader,
                                       device=device,
                                       utli=utli)

    return 0


def main(args):

    model, env = train(args)

    return model

if __name__ == '__main__':

    args = argsparser()
    main(args)