import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import rl_utils

def onehot_from_logits(logits, eps=0.01):
    ''' Generate the one hot form of the optimal action '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # Generate random actions and convert them into a one hot form
    rand_acs = torch.autograd.Variable(
        torch.eye(logits.shape[1])[[np.random.choice(range(logits.shape[1]), size=logits.shape[0])]],
        requires_grad=False).to(logits.device)
    #Using the epsilon greedy algorithm to choose which action to use
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in enumerate(torch.rand(logits.shape[0]))])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sampling from Gumbel (0,1) distribution"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Sampling from Gumbel Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """Sample from Gumbel Softmax distribution and discretize"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # Returning a solitary heat of y_hard, but with a gradient of y, we can obtain a discrete action that interacts with the environment, as well as
    # Correctly backpropagate gradients
    return y