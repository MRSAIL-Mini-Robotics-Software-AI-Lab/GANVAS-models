import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import *
from .GenBase import GenBase


class VAE(GenBase):
    # Try to pass arugments only using yaml file, you can find them in self.configs.parameter_name
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.C, self.H, self.W = self.configs.img_size
        self.n_bits = self.configs.n_bits

        self.B = self.configs.beta
        self.features = self.configs.features

        self.encode = nn.Sequential(
            nn.Linear(self.H*self.W*self.C, self.features ** 2),
            nn.ReLU(),
            nn.Linear(self.features ** 2, self.features * 2)
        )

        self.decode = nn.Sequential(
            nn.Linear(self.features, self.features ** 2),
            nn.ReLU(),
            nn.Linear(self.features ** 2, self.H*self.W*self.C),
            nn.Sigmoid(),
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

        mu_logvar = self.encode(
            x.view(-1, self.H*self.W*self.C)).view(-1, 2, self.features)
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

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)
        return mu + (eps * std)

    def samplse(self, num_to_gen: int):
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
