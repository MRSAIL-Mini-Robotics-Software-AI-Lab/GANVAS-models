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
        x = x
        sigmas = self.sigmas
        labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
        used_sigmas = sigmas[labels].view(x.shape[0], *([1] * len(x.shape[1:])))
        noise = torch.randn_like(x) * used_sigmas
        perturbed_samples = x + noise
        target = - 1 / (used_sigmas ** 2) * noise
        scores = self.forward(perturbed_samples, used_sigmas)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** 2

        return loss.mean(dim=0)

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