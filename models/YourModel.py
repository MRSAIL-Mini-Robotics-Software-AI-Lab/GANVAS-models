import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from .GenBase import GenBase

class YourModel(GenBase):
    def __init__(self, *args, **kwargs): # Try to pass arugments only using yaml file, you can find them in self.configs.parameter_name
        super().__init__(*args, **kwargs)
        C,H,W = self.configs.img_size
        self.n_bits = self.configs.n_bits
        self.configs.n_blocks

    
    def forward(self, x):
        '''
        Forward pass through the masked autoregressive model

        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, C, H, W)
        
        Returns
        -------
        '''
    
    def log_prob(self,x):
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
    
    def loss(self, x):
        '''
        Calculates the loss (usually negative log likelihood) given a set of inputs

        Parameters
        ----------
        inputs: torch.FloatTensor, shape = (batch_size, num_channels, height, width)

        Returns
        -------
        torch float representing total or average loss
        '''

    def sample(self, num_to_gen: int):
        '''
        Samples new images

        Parameters
        ----------
        num_to_gen: int
            Number of images to generate
        
        Returns
        -------
        np.ndarray, shape = (num_to_gen, num_channels, height, width)
        All values must be floats between 0 and 1
        '''