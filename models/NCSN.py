import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.distributions import Uniform, Normal
from torch.utils.data import Dataset, DataLoader

from .utils import *
from .NCSN_layers import *
from .GenBase import GenBase

class NCSN(GenBase):
    def __init__(self, *args, **kwargs): # Try to pass arugments only using yaml file, you can find them in self.configs.parameter_name
        super().__init__(*args, **kwargs)
        C,H,W = self.configs.img_size
        self.n_bits = self.configs.n_bits
        self.ref_net = RefineNet(C, self.configs.n_features)
        sigmas = gen_sigmas(self.configs.n_sigs, self.configs.max_sigma, self.configs.min_sigma)
        self.register_buffer('sigmas',sigmas)

    
    def forward(self, x, sigmas=None):
        '''
        Forward pass through the masked autoregressive model

        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, C, H, W)
        
        Returns
        -------
        '''
        x = self.ref_net.forward(x)

        if sigmas is not None:
            x = x / sigmas

        return x
    
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
        pass
    
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
        sigmas = self.sigmas
        C,H,W = self.configs.img_size
        device = next(self.parameters()).device
        b_size = x.shape[0]
        n_s = 2
        n_sig = len(sigmas)

        total_loss = 0
        sigmas = torch.tensor(sigmas).type(torch.FloatTensor).reshape(1,1,-1,1).to(device)
        noise = torch.normal(0,1,(b_size,n_s,n_sig,C)).to(device)

        x_bar = x.reshape(b_size,1,1,C) + sigmas*noise
        sigmas_val = torch.arange(0,n_sig).type(torch.FloatTensor).reshape(1,1,-1,1).to(device)
        sigmas_bs = torch.ones((b_size,n_s,n_sig,1)).to(device)*sigmas_val
        sigmas_bs = sigmas_bs.reshape(-1,1)
        pred = self.forward(x_bar.reshape(b_size*n_s*n_sig,C,H,W),sigmas_bs).reshape(b_size,n_s,n_sig,C)
        loss = sigmas*pred + noise
        loss_sq = loss*loss
        loss = torch.sum(loss_sq.reshape(b_size,-1),dim=1)
        total_loss = torch.mean(loss)
        return 0.5*total_loss/n_sig/n_s

    @torch.no_grad()
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
        sigmas = self.sigmas
        epsilon = float(self.configs.epsilon)
        T = self.configs.steps_per_sig
        batch_size = num_to_gen
        start_dist = Uniform(0,1)
        dim = self.configs.img_size
        L = sigmas.shape[0]
        device = next(self.parameters()).device
        
        x = start_dist.sample((batch_size,*dim)).to(device)
        for l in range(L):
            sigma_i = sigmas[l]
            alpha = epsilon*(sigma_i/sigmas[-1])**2
            sigs = torch.ones((batch_size,1,1,1)).to(device)*sigma_i
            for _ in range(T):
                z_t = torch.normal(0,1,(batch_size,*dim)).to(device)
                x = x + alpha*self.forward(x,sigs).reshape(batch_size,*dim) + torch.sqrt(alpha*2)*z_t
        
        x = np.clip(x.cpu().numpy(),0,1)
        return x