import torch
from torch import nn


class ActNorm(nn.Module):
    # for marginal calculations (i.e. log, div by zero, ..etc)
    EPSILON = 1e-6

    def __init__(self, in_channel):
        '''
        ActNorm class Initializer

        Parameters
        ----------
        in_channels : int
          channels of the input (image)
        '''
        super(ActNorm, self).__init__()

        self.bias = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.log_scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.is_initialized = False

    def init(self, x):
        '''
        initialize actnorm layer

        Parameters
        ----------
        x: torch.tensor
          input (first) batch of the training set
        '''
        with torch.no_grad():
            batch_size = x.shape[1]
            flattened = x.clone().permute(1, 0, 2, 3).reshape(batch_size, -1)

            # calculating mean across channels
            mean = flattened.mean(dim=1).unsqueeze(1)\
                                        .unsqueeze(2)\
                                        .unsqueeze(3).permute(1, 0, 2, 3)

            # calculatiung std across channels
            std = flattened.std(dim=1).unsqueeze(1)\
                .unsqueeze(2)\
                .unsqueeze(3).permute(1, 0, 2, 3)

            self.bias.data.copy_(-mean)
            self.log_scale.data.copy_(-1*torch.log(std + self.EPSILON))

            # mark the intialization flag
            self.is_initialized = True

    def forward(self, x, direction=0):
        '''
        calculates the actnorm

        Parameters
        ----------
        x : torch.tensor
          input (first) batch of the training set

        Returns
        -------
        actnorm : torch.Tensor
        '''
        _, _, height, width = x.shape
        x = x.type(torch.float32)
        device = next(self.parameters()).device

        if not self.is_initialized:
            self.init(x)

        # calculating log det
        scale = torch.exp(self.log_scale)
        log_det = width * height * torch.sum(self.log_scale, dim=(1, 2, 3))

        if not direction:  # direction is forward
            return x*scale + self.bias, log_det.reshape(-1)*torch.ones((x.shape[0], 1)).to(device)

        # direction is reverse
        return (x - self.bias)/scale, -log_det.reshape(-1)*torch.ones((x.shape[0], 1)).to(device)
