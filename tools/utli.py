import pandas as pd
import os
import numpy as np
import torch


# writer = SummaryWriter()
writer = []


results4loss = {
    'ppo_value': 0.0,
    'ppo_loss': 0.0,
    'ppo_entropy': 0.0,
    'bc_loss': 0.0,
    'gail_loss': 0.0,
    'gail_gamma': 0.0,
    'steer_std': 0.0,
    'throttle_std': 0.0
}


def recordLossResults(results, time_step):
    results4loss['ppo_value'] = results[0]
    results4loss['ppo_loss'] = results[1]
    results4loss['ppo_entropy'] = results[2]
    results4loss['bc_loss'] = results[3]
    results4loss['gail_loss'] = results[4]
    results4loss['gail_gamma'] = results[5]
    results4loss['steer_std'] = results[6]
    results4loss['throttle_std'] = results[7]

    write2tensorboard(results=results4loss, time_step=time_step)


results4Disloss = {
    'dis_total_loss': 0.0,
    'dis_policy_reward': 0.0,
    'dis_expert_reward': 0.0,
    'dis_loss': 0.0,
    'dis_gp': 0.0,
    'expert_loss': 0.0,
    'policy_loss': 0.0,
    'disc_pre_loss': 0.0,
    'expert_pre_reward': 0.0,
    'policy_pre_reward': 0.0,
}


def recordDisLossResults(results, time_step):
    results4Disloss['dis_total_loss'] = results[0]
    results4Disloss['dis_policy_reward'] = results[1]
    results4Disloss['dis_expert_reward'] = results[2]
    results4Disloss['dis_loss'] = results[3]
    results4Disloss['dis_gp'] = results[4]
    results4Disloss['expert_loss'] = results[5]
    results4Disloss['policy_loss'] = results[6]
    results4Disloss['disc_pre_loss'] = results[7]
    results4Disloss['expert_pre_reward'] = results[8]
    results4Disloss['policy_pre_reward'] = results[9]

    write2tensorboard(results=results4Disloss, time_step=time_step)


results4train_gail = {
    'Train reward': 0.0,
    'Train steps': 0,
    'Expert reward': 0.0,
    'Eval steps': 0.0,
    'Eval reward': 0.0
}
def recordTrainResults_gail(results, time_step):
    results4train_gail['Train reward'] = results[0]
    results4train_gail['Train steps'] = results[1]
    results4train_gail['Expert reward'] = results[2]
    results4train_gail['Eval steps'] = results[3]
    results4train_gail['Eval reward'] = results[4]

    write2tensorboard(results=results4train_gail, time_step=time_step)


def write2tensorboard(results, time_step):
    titles = results.keys()
    for title in titles:
        writer.add_scalar(title, results[title], time_step)


def Log_save_name4gail(cl_args):

    save_name = cl_args.algo + '_' + cl_args.env_name + \
                '_seed_{}_num_trajs_{}_subsample_frequency_{}_gail_{}_{}'\
                    .format(cl_args.seed,
                            cl_args.num_trajs,
                            cl_args.subsample_frequency,
                            cl_args.gail_batch_size,
                            cl_args.gail_epoch
                            )
    return save_name


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
