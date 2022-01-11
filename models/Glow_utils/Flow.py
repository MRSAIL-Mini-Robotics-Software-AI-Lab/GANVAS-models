import torch

from torch import nn
import torch.nn.functional as F

from . import *


class Flow(nn.Module):
    def __init__(self, in_channel, filters=512):
        '''
        Flow class Initializer

        Parameters
        ----------
        in_channels : int
          channels of the input (image)
        filters : int
          filters input to the affineCoupling layer
        '''
        super(Flow, self).__init__()
        self.actnorm = ActNorm(in_channel=in_channel)
        self.invConv = InvConv2d(in_channel=in_channel)
        self.affineCoupling = AffineCoupling(in_channel, filters)

    def forward(self, x, direction=0):
        '''
        Forward function to for the flow step (Actnorm + 
        InvConv + AffineCoupling)

        Parameters
        ----------
        x : torch.tensor
          shape (batch_size, in_channels, height, width)
        direction : int
          0 means forward convolution
          1 means inverse convolution

        Returns
        -------
        out : torch.tensor
        logdet : int
        '''
        log_det1, log_det2, log_det3 = 0, 0, 0
        if not direction:
            x, log_det1 = self.actnorm(x, direction)
            x, log_det2 = self.invConv(x, direction)
            x, log_det3 = self.affineCoupling(x, direction)
        else:
            x, log_det1 = self.affineCoupling(x, direction)
            x, log_det2 = self.invConv(x, direction)
            x, log_det3 = self.actnorm(x, direction)

        logdet = log_det1 + log_det2 + log_det3

        return x, logdet
