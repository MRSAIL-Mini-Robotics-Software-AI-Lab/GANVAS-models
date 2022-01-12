import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from .GenBase import GenBase
from .convlayers import maskedConv2d,residualBlock

class pixelCNN(GenBase):
    def __init__(self, *args, **kwargs): # Try to pass arugments only using yaml file, you can find them in self.configs.parameter_name
        super(pixelCNN,self).__init__(*args, **kwargs)
        C,H,W = self.configs.img_size
        self.n_bits = self.configs.n_bits
        self.img_channels = C
        self.img_height = H
        self.img_width = W
        self.quantization_levels = np.power(2,self.n_bits)
        self.model = [
            maskedConv2d(self.img_channels, 2*self.configs.h, self.configs.start_k, maskType="A", imgChannels=self.img_channels, padding="same")]
        for i in range(self.configs.n_blocks):
            self.model.append(residualBlock(self.configs.h, self.configs.residual_k, self.img_channels))
        self.model.append(nn.ReLU())
        self.model.extend([maskedConv2d(2*self.configs.h, self.configs.h, 1, maskType = "B", imgChannels = self.img_channels), nn.ReLU()])
        self.model.extend([maskedConv2d(self.configs.h, self.quantization_levels * self.img_channels, 1, maskType = "B", imgChannels = self.img_channels)])
        self.model = nn.Sequential(*self.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cross_entropy = nn.NLLLoss()
    
    def forward(self, x):
        '''
        Forward pass through the masked autoregressive model

        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, C, H, W)
        
        Returns
        -------
        '''
        # y shape: minibatch,quantization_level,channels,img_heigh,img_weidth
        y = self.model.forward(x)
        y = y.reshape(-1, self.quantization_levels, self.img_channels, self.img_height, self.img_width)
        y = nn.LogSoftmax(1)(y)
        return y
    
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
        
        q = x.clone().type(torch.LongTensor).to(self.device)
        l = self.cross_entropy(self.forward(x),q)
        return l

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
        samples =torch.zeros(num_to_gen, self.img_channels, self.img_height, self.img_width)
        with torch.no_grad():
            for n in range(num_to_gen):
                inp = torch.zeros(1, self.img_channels, self.img_height, self.img_width).float().to(self.device)
                for r in range(self.img_height):
                    for c in range(self.img_width):
                        for i in range(self.img_channels):
                            prob = torch.exp(self.forward(inp)[0,:, i, r, c])
                            p = torch.multinomial(prob,1).float().item()
                            inp[0, i, r, c] = p 
                samples[n,:,:,:]=inp[0]/(self.quantization_levels-1)
        return torch.FloatTensor(samples).cpu().detach().numpy()