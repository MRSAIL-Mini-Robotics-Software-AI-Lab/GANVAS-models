import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.distributions import Uniform, Normal
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

def gen_sigmas(L,sig_1,sig_L):
    '''
    Generate a sequence of sigmas using a geometric sequence

    Parameters
    ----------
    L: int
        The number of sigmas to generate
    sig_1: float
        The first sigma
    sig_L: float
        The last sigma
    
    Returns
    -------
    torch.FloatTensor, shape=(L,)
        Each element is a sigma
    '''
    coef = np.power(sig_L/sig_1,1/(L-1))
    sigs = [sig_1]
    for i in range(L-1):
        sigs.append(sigs[-1]*coef)
    
    sigs = torch.tensor(sigs).float()
    return sigs

class RCU(nn.Module):
    def __init__(self,n_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.ELU(),
                                      nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
                                      nn.ELU(),
                                      nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
    
    def forward(self, x):
        out = self.layers(x) + x
        return out
class RCUBlock(nn.Module):
    def __init__(self, n_channels, n_blocks=2):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(RCU(n_channels))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks.forward(x)

class MRF(nn.Module):
    def __init__(self,in_channels, out_channel):
        super().__init__()
        convs = []
        for i in range(2):
            convs.append(nn.Conv2d(in_channels[i],out_channel, kernel_size=3, padding=1))
    
        self.convs = nn.ModuleList(convs)

    def forward(self, x1, x2):
        inp_arr = [x1, x2]
        larger_shape_idx = np.argmax([x1.shape[2],x2.shape[2]])
        out_shape = inp_arr[larger_shape_idx].shape

        for i in range(2):
            inp_arr[i] = self.convs[i].forward(inp_arr[i])
            if i != larger_shape_idx:
                inp_arr[i] = F.interpolate(inp_arr[i], out_shape[2:], mode='bilinear', align_corners=True)
        
        out = inp_arr[0] + inp_arr[1]
        return out


class CRP(nn.Module):
    def __init__(self,n_channels, n_stages):
        super().__init__()
        convs = []
        for i in range(n_stages):
            convs.append(nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1))
        
        self.convs = nn.ModuleList(convs)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.activation = nn.ELU()
    
    def forward(self, x):
        x = self.activation(x)
        out = torch.clone(x)

        for i in range(len(self.convs)):
            x = self.pool(x)
            x = self.convs[i].forward(x)
            out += x
        
        return out

class RefineBlock(nn.Module):
    def __init__(self,num_channels):
        super().__init__()
        self.rcu_1 = RCUBlock(num_channels)
        self.rcu_2 = RCUBlock(num_channels)
        self.out_rcu = RCUBlock(num_channels)

        self.mrf = MRF([num_channels,num_channels], num_channels)

        self.crp = CRP(num_channels,3)
    
    def forward(self, x1, x2):
        x1 = self.rcu_1(x1)
        x2 = self.rcu_2(x2)

        x = self.mrf(x1, x2)

        x = self.crp(x)

        x = self.out_rcu(x)

        return x

class RefineBlockStart(nn.Module):
    def __init__(self,num_channels):
        super().__init__()
        self.rcu = RCUBlock(num_channels)
        self.out_rcu = RCUBlock(num_channels)

        self.mrf = nn.Conv2d(num_channels,num_channels, kernel_size=3, padding=1)

        self.crp = CRP(num_channels,3)
    
    def forward(self,x):
        x = self.rcu(x)

        x = self.mrf(x)

        x = self.crp(x)

        x = self.out_rcu(x)

        return x

class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.norm_1 = nn.InstanceNorm2d(in_channels)
        self.norm_2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.ELU()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        out = self.norm_1(x)
        out = self.activation(out)
        out = self.conv_1(out)
        out = self.norm_2(out)
        out = self.activation(out)
        out = self.conv_2(out)

        out = self.shortcut(x) + out
        return out
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, n_units=3, downsample=True):
        super().__init__()
        blocks = [ResUnit(in_channels, out_channels, dilation=dilation)]
        for i in range(n_units-1):
            blocks.append(ResUnit(out_channels, out_channels, dilation=dilation))
        
        self.blocks = nn.ModuleList(blocks)

        if downsample:
            self.downsample = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        else:
            self.downsample = nn.Identity()
        
    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        x = self.downsample(x)
        return x

class RefineNet(nn.Module):
    def __init__(self, in_channels, n_features):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1)
        self.last_conv = nn.Conv2d(2*n_features, in_channels, kernel_size=3, padding=1)

        self.norm = nn.BatchNorm2d(2*n_features)
        self.activation = nn.ELU()

        self.res_block1 = ResBlock(n_features, 2*n_features, downsample=False)
        self.res_block2 = ResBlock(2*n_features, 2*n_features)
        self.res_block3 = ResBlock(2*n_features, 2*n_features, dilation=2)
        self.res_block4 = ResBlock(2*n_features, 2*n_features, dilation=4)


        self.refine4 = RefineBlockStart(2*n_features)
        self.refine3 = RefineBlock(2*n_features)
        self.refine2 = RefineBlock(2*n_features)
        self.refine1 = RefineBlock(2*n_features)

    def forward(self, x):
        x = self.conv_1(x)

        scale_1 = self.res_block1.forward(x)
        scale_2 = self.res_block2.forward(scale_1)
        scale_3 = self.res_block3.forward(scale_2)
        scale_4 = self.res_block4.forward(scale_3)

        out_ref_4 = self.refine4.forward(scale_4)
        out_ref_3 = self.refine3.forward(out_ref_4, scale_3)
        out_ref_2 = self.refine2.forward(out_ref_3, scale_2)
        out_ref_1 = self.refine1.forward(out_ref_2, scale_1)

        out = self.norm.forward(out_ref_1)
        out = self.activation(out)
        out = self.last_conv.forward(out)

        return out