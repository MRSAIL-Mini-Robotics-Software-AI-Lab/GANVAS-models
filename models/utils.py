import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def uniform_dist(a, b, size):
    '''
    Sample from a uniform distribution Unif(a,b)
    '''
    std_unif = torch.rand(size)
    return std_unif*(b-a)+a

def safe_log(tens, epsilon:float=1e-5):
    '''
    Safe log to prevent infinities
    '''
    return torch.log(tens+epsilon)

def sample_dist(probs):
    '''
    Sample from a given probability distribution

    Parameters
    ----------
    probs: numpy.float array, shape = (num_samples, num_values)
        Note: the sum of each row must be = 1
    '''
    num_values = probs.shape[1]
    generated = []
    for prob in probs:
        generated.append(np.random.choice(np.arange(num_values), p=prob))
    return np.array(generated)