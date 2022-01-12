import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
from .utils import *
from .GenBase import GenBase
from .convlayers import maskedConv2d,gatedBlock

class gatedBlockStack(nn.Module):
  #Stack of gatedblocks 
  def __init__(self,n_blocks,hidden_channels,kernel_size,img_channels):
    super(gatedBlockStack,self).__init__()
    self.n_blocks = n_blocks
    self.blocks = []
    for i in range(n_blocks):
      self.blocks.append(gatedBlock(in_channels = hidden_channels,out_channels = hidden_channels, kernel_size=kernel_size, imgChannels=img_channels,Masktype="B"))
    self.net = nn.Sequential(*self.blocks)
  def forward(self,v,h,skip):
    return self.net.forward((v,h,skip))

class gatedPixelCNN(GenBase):
  def __init__(self, *args, **kwargs):
    super(gatedPixelCNN,self).__init__(*args, **kwargs)
    C,H,W = self.configs.img_size
    self.n_bits = self.configs.n_bits
    self.img_channels = C
    self.img_height = H
    self.img_width = W
    self.quantization_levels = np.power(2,self.n_bits)
    self.layer1 = gatedBlock(in_channels=self.img_channels,out_channels=self.configs.h_channels,kernel_size=self.configs.kernel_size,imgChannels=self.img_channels,Masktype="A")
    self.blockList = gatedBlockStack(self.configs.n_blocks,hidden_channels=self.configs.h_channels,kernel_size=self.configs.kernel_size,img_channels=self.img_channels)
    self.layer3 = maskedConv2d(self.configs.h_channels,self.configs.h_channels,1,maskType = "B",imgChannels=self.img_channels)
    self.layer4 = maskedConv2d(self.configs.h_channels,self.quantization_levels*self.img_channels,1,maskType = "B",imgChannels=self.img_channels)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.layer1.to(self.device)
    self.blockList.to(self.device)
    self.layer3.to(self.device)
    self.layer4.to(self.device)
    self.cross_entropy = nn.NLLLoss()

  def forward(self,x):
    '''
        Forward pass through the masked autoregressive model

        Parameters
        ----------
        x: torch.FloatTensor, shape = (batch_size, C, H, W)
        
        Returns
        -------
    '''
    
    v,h,skip = self.layer1.forward((x,x,x.new_zeros((x.shape[0],self.configs.h_channels,self.img_height,self.img_width), requires_grad=True)))

    _,_,h = self.blockList.forward(v,h,skip)
    
    h = nn.ReLU()(h)
    h = self.layer3.forward(h)

    h = nn.ReLU()(h)
    h = self.layer4.forward(h)
    
    h = h.reshape(-1, self.quantization_levels, self.img_channels, self.img_height, self.img_width)
    # y shape: minibatch,quantization_level,channels,img_height,img_width
    y = nn.LogSoftmax(1)(h)
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




