import torch

from torch import nn
import torch.nn.functional as F
from torch.utils import checkpoint


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filters=256):
        super().__init__()
        in_channels = in_channel
        filter_size = filters
        self.flow_net = nn.Sequential(nn.Conv2d(in_channels, filter_size, kernel_size=(3, 3), padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(filter_size, filter_size,
                                                kernel_size=(1, 1), padding=0),
                                      nn.ReLU(),
                                      nn.Conv2d(filter_size, in_channels*2, kernel_size=(3, 3), padding=1))

        self.flow_net[-1].weight.data *= 0
        self.flow_net[-1].bias.data *= 0

        self.mask = torch.ones(in_channels, 1, 1)

        self.mask[:in_channels//2] *= 0

        self.mask = nn.Parameter(self.mask.reshape(1, in_channels, 1, 1))
        self.mask.requires_grad = False

        self.scale = nn.Parameter(torch.zeros(1))
        self.scale_shift = nn.Parameter(torch.zeros(1))

    def forward(self, x, direction=0):
        '''
        forward function of affine coupling layer

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
        x_ = x*self.mask
        out = checkpoint.checkpoint(self.flow_net, x_)
        log_scale, t = torch.chunk(out, 2, dim=1)
        log_scale = self.scale*torch.tanh(log_scale) + self.scale_shift

        t = t * (1.0 - self.mask)
        log_scale = log_scale * (1.0 - self.mask)
        if direction:
            z = (x - t) / torch.exp(log_scale)
            log_det_jacobian = -1*log_scale
        else:
            z = x * torch.exp(log_scale) + t
            log_det_jacobian = log_scale

        return z, torch.sum(log_det_jacobian, dim=(1, 2, 3)).reshape(-1, 1)
