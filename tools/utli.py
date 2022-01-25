import pandas as pd
import numpy as np


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
    'dis_ct': 0.0,
    'expert_loss': 0.0,
    'policy_loss': 0.0,
    'disc_pre_loss': 0.0,
    'expert_pre_reward': 0.0,
    'policy_pre_reward': 0.0,
    'disc_after_loss': 0.0,
    'expert_after_reward': 0.0,
    'policy_after_reward': 0.0,
}


def recordDisLossResults(results, time_step):
    results4Disloss['dis_total_loss'] = results[0]
    results4Disloss['dis_policy_reward'] = results[1]
    results4Disloss['dis_expert_reward'] = results[2]
    results4Disloss['dis_loss'] = results[3]
    results4Disloss['dis_gp'] = results[4]
    results4Disloss['dis_ct'] = results[5]
    results4Disloss['expert_loss'] = results[6]
    results4Disloss['policy_loss'] = results[7]
    results4Disloss['disc_pre_loss'] = results[8]
    results4Disloss['expert_pre_reward'] = results[9]
    results4Disloss['policy_pre_reward'] = results[10]
    results4Disloss['disc_after_loss'] = results[11]
    results4Disloss['expert_after_reward'] = results[12]
    results4Disloss['policy_after_reward'] = results[13]

    write2tensorboard(results=results4Disloss, time_step=time_step)


results4train_gail = {
    'Train reward': 0.0,
    'Train steps': 0,
    'Expert reward': 0.0,
    'Eval steps': 0.0,
    'Eval reward': 0.0,
    'disc_eval_loss': 0.0,
    'expert_eval_reward': 0.0,
    'policy_eval_reward': 0.0
}
def recordTrainResults_gail(results, time_step):
    results4train_gail['Train reward'] = results[0]
    results4train_gail['Train steps'] = results[1]
    results4train_gail['Expert reward'] = results[2]
    results4train_gail['Eval steps'] = results[3]
    results4train_gail['Eval reward'] = results[4]
    results4train_gail['disc_eval_loss'] = results[5]
    results4train_gail['expert_eval_reward'] = results[6]
    results4train_gail['policy_eval_reward'] = results[7]

    write2tensorboard(results=results4train_gail, time_step=time_step)


def record_routes_rewards(routes_rewards, time_step):
    routes_metrics = {}
    for route_idx in routes_rewards.keys():
        metric_id = 'route_{:0>2d}_max_reward'.format(route_idx)
        routes_metrics[metric_id] = np.max(routes_rewards[route_idx])
        metric_id = 'route_{:0>2d}_min_reward'.format(route_idx)
        routes_metrics[metric_id] = np.min(routes_rewards[route_idx])

    write2tensorboard(routes_metrics, time_step)


def write2tensorboard(results, time_step):
    titles = results.keys()
    for title in titles:
        writer.add_scalar(title, results[title], time_step)


def Log_save_name4gail(run_params):

    save_name = run_params['algo'] + '_' + run_params['env_name'] + \
                '_seed_{}_gail_{}_{}'\
                    .format(run_params['seed'],
                            run_params['gail_batch_size'],
                            run_params['gail_epoch']
                            )
    return save_name


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
