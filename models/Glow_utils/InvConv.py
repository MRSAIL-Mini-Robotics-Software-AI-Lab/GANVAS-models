import torch

import numpy as np
from torch import nn
import torch.nn.functional as F
from scipy import linalg

class InvConv2d(nn.Module):
    # for marginal calculations (i.e. log, div by zero, ..etc)
    EPSILON = 1e-6

    def __init__(self, in_channel):
        '''
        InvConv2d class Initializer

        Parameters
        ----------
        in_channels : int
          channels of the input (image)
        '''
        super(InvConv2d, self).__init__()

        # computing QR decomposition
        weight = np.random.randn(in_channel, in_channel)
        q, _ = linalg.qr(weight)

        # computing PLU decomposition
        q = q.astype(np.float32)
        P, L, U = linalg.lu(q.astype(np.float32))
        s = np.diag(U)
        log_s = np.log(np.abs(s)+self.EPSILON)  # for numerical stability
        U = np.triu(U, 1)

        # masks for L, U matricies and adding diagonal
        self.register_buffer(
            "U_mask", torch.from_numpy(np.triu(np.ones_like(U),1)))
        self.register_buffer("L_mask", self.U_mask.T)
        self.register_buffer("diag", torch.eye(L.shape[0]))
        self.register_buffer('P', torch.from_numpy(P))
        self.register_buffer('s_signs', torch.sign(torch.from_numpy(s)))

        # convert to torch tensors
        self.log_s = nn.Parameter(torch.from_numpy(log_s))
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

    def __getConvWeight(self, direction=0):
        '''
        Calculates the weight matrix of the invertible convolution
        using LU Decomposition

        Returns
        -------
        Weight matrix : torch.tensor
          has the shape in_channels, in_channels, 1, 1
        logdet : int
        '''
        p = self.P
        l = (self.L * self.L_mask + self.diag)
        u = ((self.U * self.U_mask) +
             torch.diag(torch.exp(self.log_s)*self.s_signs))
        if direction:
            p = torch.inverse(p)
            l = torch.inverse(l)
            u = torch.inverse(u)
            weight = u@l@p
            return weight.unsqueeze(2).unsqueeze(3)
        weight = p@l@u

        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, x, direction=0):
        '''
        forward function to calculate the 1x1 convolution

        Parameters
        ----------
        x : torch.tensor
          shape (batch_size, in_channels, height, width)
        direction : int
          0 means forward convolution
          1 means inverse convolution

        out : torch.tensor
        logdet : int
        '''
        _, channels, height, width = x.shape
        x = x.type(torch.float32)
        device = next(self.parameters()).device

        # calculating conv weight window
        weight = self.__getConvWeight(direction)

        # log det
        logdet = height * width * \
            torch.sum(self.log_s) * \
            torch.ones((x.shape[0], 1)).to(device)

        if not direction:
            out = F.conv2d(x, weight)
            # print("CONV_LOG_DET", logdet)
            return out, logdet
        out = F.conv2d(input=x, weight=weight)
        return out, -logdet
