import torch

import numpy as np
from torch import nn
import torch.nn.functional as F

from . import *

class GlowLevel(nn.Module):
    def __init__(self, in_channel, filters=512, n_levels=1, n_steps=2):
      '''
      Iniitialized Glow Layer

      Parameters
      ----------
      in_channel : int
        number of input channels
      filters : int
        number of filters in affine coupling layer
      n_levels : int
        number of Glow layers
      n_steps : int
        number of flow steps
      '''
      super(GlowLevel, self).__init__()
    
      # init flow layers
      self.flowsteps = nn.ModuleList([Flow(in_channel*4, filters = filters)
                      for _ in range(n_steps)])
      
      # init Glow levels
      if(n_levels > 1):
        self.nextLevel = GlowLevel(in_channel = in_channel * 2,
                                    filters = filters,
                                    n_levels = n_levels-1,
                                    n_steps = n_steps)
      else:
        self.nextLevel = None
      
    def forward(self, x, direction = 0):
      '''
      forward function for each glow level

      Parameters
      ----------
      x : torch.tensor
        input batch
      direction : int
        0 means forward
        1 means reverse

      Returns
      -------
      x : torch.tensor
        output of the glow layer
      logdet : float
        the log-determinant term
      '''
      sum_logdet = 0
      x = self.squeeze(x)
    
      if not direction: # direction is forward
        for flowStep in self.flowsteps:
          x, log_det = flowStep(x, direction=direction)
          sum_logdet += log_det

      if self.nextLevel is not None:
        x, x_split = x.chunk(2, 1)
        x, log_det = self.nextLevel(x, direction)
        sum_logdet += log_det
        x = torch.cat((x, x_split), dim = 1)
      
      if direction: # direction is reverse
        for flowStep in reversed(self.flowsteps):
          x, log_det = flowStep(x, direction)
          sum_logdet += log_det
        
      x = self.unsqueeze(x)

      return x, sum_logdet

    def squeeze(self, x):
        """
        Quadruples the number of channels of the input
        this is done to increase the spatial information volume for the image channels

        Parameters
        ----------
        x : torch.Tensor
          batch of input images with shape
          batch_size, channels, height, width
        
        Returns
        -------
        x : torch.tensor
          squeezed input with shape
            batch_size, channels * 4, height//2, width//2
        """
        batch_size, channels, h, w = x.size()
        x = x.view(batch_size, channels, h // 2, 2, w //2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous() # to apply permutation from the glow paper
        x = x.view(batch_size, channels * 4, h // 2, w // 2)
        return x

    def unsqueeze(self, x):
        """
        Unsqueeze the image by dividing the channels by 4 and
        reconstructs the squeezed image 

        Parameters
        ----------
        x : torch.Tensor
          batch of input images with shape
          batch_size, channels * 4, height//2, width//2
        
        Returns
        -------
        x : torch.tensor
          unsqueezed input with shape
          batch_size, channels, height, width
        """
        batch_size, channels, h, w = x.size()
        x = x.view(batch_size, channels // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_size, channels // 4, h * 2, w * 2)
        return x