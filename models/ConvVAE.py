import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from .GenBase import GenBase


class ConvVAE(GenBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.C, self.H, self.W = self.configs.img_size

        self.n_bits = self.configs.n_bits

        self.B = self.configs.B
        self.features = self.configs.features

        self.filter_num = self.configs.filter_num
        self.B = self.configs.B
        self.features = self.configs.features

        H = int(self.H*self.W*self.filter_num/4)

        class Reshape(nn.Module):
            def __init__(self, filter_num, h, w):
                super().__init__()
                self.filter_num = filter_num
                self.h = h
                self.w = w

            def forward(self, x):
                return x.view(-1, self.filter_num, self.h, self.w)

        self.encode = nn.Sequential(
            nn.Conv2d(self.C, self.filter_num, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.filter_num, self.filter_num *
                      2, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.filter_num*2, self.filter_num*4, 3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(H, self.features*2)
        )

        self.decode = nn.Sequential(
            nn.Linear(self.features, H),
            Reshape(self.filter_num*4, self.H//4, self.W//4),
            nn.ConvTranspose2d(self.filter_num*4,
                               self.filter_num*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.filter_num*2, self.filter_num,
                               3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.filter_num, self.C, 3),
            nn.Sigmoid()
        )

        self.criterion = torch.nn.BCELoss(reduction='sum')

    def forward(self, x):
        '''
        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, C, H, W)

        Returns
        -------
        '''

        mu_logvar = self.encode(x).view(-1, 2, self.features)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

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
        target, mu, logvar = self.forward(x)

        recon_loss = self.criterion(target.view(-1, self.C, self.H, self.W), x)
        KLDivergence = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

        return recon_loss + self.B * KLDivergence

    def reparameterize(self, mu, logvar):
        """
        Parameters
        ----------
        mu: torch.FloatTensor, shape = (batch_size, dimensions)
            Mean of the encoder

        log_var: torch.FloatTensor, shape = (batch_size, dimensions)
            log variance from the encoder

        Returns
        -------
        sample: torch.FloatTensor, shape = (batch_size, dimensions)
            Sampled z from the input distribution + some noise 
        """

        device = next(self.parameters()).device

        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std).to(device)
            return mu + (eps * std)
        else:
            return mu

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

        device = next(self.parameters()).device

        z = torch.randn((num_to_gen, self.features)).to(device)

        sample = self.decode(z).view(-1, self.C, self.H,
                                     self.W).detach().cpu().numpy()

        return sample
