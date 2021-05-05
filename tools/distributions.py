import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, activation=None):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = torch.zeros(num_outputs)
        self.activation = activation

    def update_entropy(self, entropy):
        self.logstd = self.logstd.fill_(entropy / self.logstd.shape[0] - (0.5 + 0.5 * math.log(2 * math.pi)))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        if self.activation is not None:
            action_mean = self.activation(action_mean)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd.to(action_mean.device)
        return FixedNormal(action_mean, action_logstd.exp())
