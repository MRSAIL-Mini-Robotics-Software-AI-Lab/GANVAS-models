import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from .GenBase import GenBase
from .Glow_utils import *


class Glow(GenBase):
    # Try to pass arugments only using yaml file, you can find them in self.configs.parameter_name
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.C, self.H, self.W = self.configs.img_size
        self.n_bits = self.configs.n_bits
        self.filters = self.configs.n_filters
        self.n_levels = self.configs.n_levels
        self.n_steps = self.configs.n_steps
        self.flows = GlowLevel(in_channel=self.C,
                               filters=self.filters,
                               n_levels=self.n_levels,
                               n_steps=self.n_steps)

    def forward(self, x, direction=0):
        '''
        Forward pass through the masked autoregressive model

        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, C, H, W)
        direction: int, indicate the direction
          0 : forward
          1 : inverse

        Returns
        -------
        out: torch.FloatTensor, shape = (batch_size, C, H, W)
        '''
        # print("SHAPE",x.shape)
        # return
        if not direction:
            # Possibly check if image is within [0,1] or [0,255] range
            x, Mloga = self.dequantize(x)
            x, sum_logdet = self.flows(x)
            sum_logdet += Mloga

        else:
            x, sum_logdet = self.flows(x, direction)
            x, Mloga = self.quantize(x)
            sum_logdet += Mloga

        return x, sum_logdet

    def dequantize(self, x, n_bins=2**8, n_bits=8):
        '''
        Dequantizes discrete input x where each color can be one of n_bins values
        by adding uniform noise

        Parameters
        ----------
        x : torch.tensor
          shape (batch_size, in_channels, height, width)
        n_bits_x : int
          num of bits representing a single color
        n_bins : int
          num of discrete bins

        x : torch.tensor
          dequantized image
        logdet : float
        '''

        x = x.type(torch.float32)

        x = x + uniform_dist(0, 1, x.shape).to(x.device)

        alpha = 0.05
        # Logit Trick
        factor = 1/2
        x = x*factor
        x = (1-2*alpha)*x + alpha
        log_x = safe_log(x)
        log_1_x = safe_log(1-x)
        new_x = log_x - log_1_x
        log_det_jac = np.log((1-2*alpha)*factor) - log_x - log_1_x

        return new_x, torch.sum(log_det_jac, dim=(1, 2, 3)).reshape(-1, 1)

    def quantize(self, x):
        alpha = 0.05
        x = torch.sigmoid(x)  # Inverse of logit
        factor = 2
        new_x = factor*(x-alpha)/(1-2*alpha)
        log_x = safe_log(x)
        log_1_x = safe_log(1-x)
        log_det_jac = log_x + log_1_x - np.log(1-2*alpha) + np.log(factor)
        return new_x/2, torch.sum(log_det_jac, dim=(1, 2, 3)).reshape(-1, 1)

    def _prior_ll(self, z):
        '''
        Prior log probability of z
        Its an isotropic unit gaussian
        '''

        single_feature_ll = -0.5 * (np.log(2*np.pi) + z**2)
        zprior_ll = single_feature_ll.flatten(1).sum(-1)
        return zprior_ll

    def log_prob(self, x):
        '''
        Estimate the nats per dim of given samples

        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, C, H, W)

        Returns
        -------
        torch.FloatTensor, shape = (batch_size)
            Each element is the nats per dim of the respective sample from x
        '''
        x = self.dequantize(x)
        return torch.sum(-torch.log(self.forward(x)), axis=(1, 2, 3))

    def loss(self, x):
        '''
        Calculates the loss (usually negative log likelihood) given a set of inputs

        Parameters
        ----------
        inputs: torch.FloatTensor, shape = (batch_size, num_channels, height, width)
        sum_logdet : torch.Tensor sum of the log determinants

        Returns
        -------
        torch float representing total or average loss
        '''
        # print("LOGDET",sum_logdet)
        z, sum_logdet = self.forward(x)
        num_dims = 20*20
        loss = -(self._prior_ll(z)+sum_logdet).mean()/num_dims
        #loss =  -(self._prior_ll(z)).mean()
        return loss

    def sample(self, num_to_gen: int, temprature: int = 0.7):
        '''
        Samples new images

        Parameters
        ----------
        num_to_gen: int
            Number of images to generate

        temprature: int
            dispertion factor of the random samples

        Returns
        -------
        np.ndarray, shape = (num_to_gen, num_channels, height, width)
        All values must be floats between 0 and 1
        '''
        device = device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        z = torch.randn((num_to_gen, self.C, self.H, self.W)
                        ).to(device) * temprature
        out, _ = self.forward(z, direction=1)

        # convert to numpy array
        out = out.cpu().detach().numpy()
        out = np.round(np.clip(out, 0, 1))

        return out
