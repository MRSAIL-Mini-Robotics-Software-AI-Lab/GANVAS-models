
import torch
import torch.nn as nn
import numpy as np

class maskedConv2d(nn.Conv2d):
  '''
  Masked 2d covolutions, mainly used for pixelCNN layers.
  Also, it's used in the gated-Pixel-CNN for (1,1) convultions where,
  we want to have masking between R,G,B channels as described in the documentation.
  '''
  def __init__(self,*args,maskType,imgChannels,**kwargs):
    super(maskedConv2d,self).__init__(*args,**kwargs)
    
    out_channels, in_channels, height, width = self.weight.size()

    mask = np.zeros((out_channels,in_channels,height,width))

    cx = width//2
    cy = height//2
    mask[ : , : , :cy , : ] = 1
    mask[ : , : , cy ,  :cx+1 ] = 1

    '''
    Get Indexes which should be masked. 
    If data channels is R,G,B 
    R can access --> None
    G can access --> R
    B can access --> R,G

    Channels are grouped in truplets (R,G,B,R,G,B)
    '''
    def cmask(out_c, in_c):
      a = (np.arange(out_channels) % imgChannels == out_c)[:, None]
      b = (np.arange(in_channels) % imgChannels == in_c)[None, :]
      return a * b

    for o in range(imgChannels):
      for i in range(o + 1, imgChannels):
        mask[cmask(o, i), cy, cx] = 0

    if maskType == 'A':
      for c in range(imgChannels):
        mask[cmask(c, c), cy, cx] = 0

    mask = torch.from_numpy(mask).float()

    self.register_buffer('mask',mask)

  def forward(self,x):
    self.weight.data *= self.mask
    x = super(maskedConv2d,self).forward(x)
    return x

class vConv2d(nn.Conv2d):
  '''
  Vertical convolution, implemented by using kernel size of (K\\2,k),
  and padding the image, and cropping to preserve the spatial dimensionality.
  This is an alternative for masking, notice we didn't mask the weights at all.
  '''
  def __init__(self,in_channels,out_channels,kernel_size):
    self.cropped_kernel_size = kernel_size//2+1
    self.k = kernel_size
    left_right_padding = self.k//2
    super(vConv2d,self).__init__(in_channels,out_channels,(self.cropped_kernel_size,kernel_size),bias = True,padding = (self.cropped_kernel_size,left_right_padding))
  def forward(self,x):
    x = super(vConv2d,self).forward(x)
    v = x[:,:,1:-self.cropped_kernel_size,:]
    # Shifting the output to pe passed for the horizontal stack.
    shifted_v = x[:,:,:-self.cropped_kernel_size-1,:]
    return shifted_v,shifted_v

class hConv2d(nn.Conv2d):
  ''' 
  Horizontal convolution, also implemented with padding and cropping,
  using kernel size of (1,k\\2), and padding the input and then cropping the output.
  Horizontal convolution differs in the first layer from the following layers,
  so for type A we dont want to condition on the current value, however 
  for type B we can condition on the current pixel, this is only possible 
  due to the "masking" done in the first layer
  '''
  def __init__(self,Masktype,in_channels,out_channels,kernel_size):
    self.cropped_kernel_size = kernel_size//2+1
    if Masktype == "A":
      self.pad_and_crop = kernel_size//2+1
    elif Masktype == "B":
      self.pad_and_crop  = kernel_size//2
    self.k = kernel_size
    super(hConv2d,self).__init__(in_channels,out_channels,(1,self.cropped_kernel_size),bias = True,padding = (0,self.pad_and_crop))

  def forward(self,x):
    b,c,h,w = x.shape
    x = super(hConv2d,self).forward(x)
    return x[:,:,:,:w]

class vstack(nn.Module):
  '''
  V-stack, implementation, applying the gated activation using the same weights
  and splitting the channels before the gated function. Also, we added a residual connection
  the paper mentioned that it didn't improve the performance.    
  '''
  def __init__(self,in_channels,out_channels,kernel_size,imgChannels,do_residual):
    super(vstack, self).__init__()
    self.do_residual = do_residual
    self.out_channels = out_channels
    self.vlayer = vConv2d(in_channels,2*out_channels,kernel_size)
    self.v_to_h = maskedConv2d(2*out_channels,2*out_channels,1,maskType="B",imgChannels=imgChannels)
    if do_residual:
      self.residual = maskedConv2d(out_channels,out_channels,1,maskType="B",imgChannels=imgChannels)
  def forward(self,x):
    v,shifted_v = self.vlayer.forward(x)
    to_h = self.v_to_h.forward(shifted_v)
    v1,v2 = torch.split(v, self.out_channels, dim=1)
    finalV = torch.sigmoid(v1) * torch.tanh(v2)
    if self.do_residual:
      finalV =finalV + self.residual.forward(finalV)
    return finalV,to_h

class hstack(nn.Module):
  '''
  H-stack implementation, applying convolution and adding information from the vertical
  stack and then splitting the channels for gating activation function. Also, 
  There is a residual connection for the H stack as well.
  '''
  def __init__(self,in_channels,out_channels,kernel_size,imgChannels,Masktype):
    super(hstack, self).__init__()
    self.out_channels = out_channels
    self.do_residuals = Masktype == "B"
    self.hlayer = maskedConv2d(in_channels,2*out_channels,(1,kernel_size),padding = (0,kernel_size//2),maskType=Masktype,imgChannels=imgChannels)
    if self.do_residuals:
      self.residual = maskedConv2d(out_channels,out_channels,1,maskType="B",imgChannels=imgChannels)
  def forward(self,x,from_v):
    h0 = self.hlayer.forward(x)
    h0 =h0 + from_v
    h1,h2 = torch.split(h0, self.out_channels, dim=1)
    h = torch.sigmoid(h1) * torch.tanh(h2)
    if self.do_residuals:
      res_out = self.residual.forward(h)
      h = torch.add(x,res_out)
    return h

class gatedBlock(nn.Module):
  '''
  Bringing every thing together the H-stack and the V-stack, and adding 
  a skip conection from every block to the output layer.
  '''
  def __init__(self,in_channels,out_channels,kernel_size,imgChannels,Masktype):
    super(gatedBlock, self).__init__()
    self.hstack = hstack(in_channels,out_channels,kernel_size,imgChannels,Masktype)
    self.vstack = vstack(in_channels,out_channels,kernel_size,imgChannels,do_residual=True)
    self.do_skip = Masktype == "B"
    self.skip = maskedConv2d(out_channels,out_channels,1,maskType="B",imgChannels=imgChannels)
  def forward(self,x):
    v,h,skip = x
    v,to_h = self.vstack.forward(v)
    h = self.hstack.forward(h,to_h)
    skip = skip+self.skip.forward(h)
    return v,h,skip

class residualBlock(nn.Module):
  '''
  Residual block, that was only used in pixelCNN as described in the documentation
  '''
  def __init__(self,no_in_f,kernel_size,imgChannels):
    super(residualBlock, self).__init__()
    self.conv1 = nn.Conv2d(2*no_in_f,no_in_f,1) # 1*1 Conv
    self.masked = maskedConv2d(no_in_f,no_in_f,kernel_size,maskType = "B",imgChannels=imgChannels,padding="same")
    self.conv2 = nn.Conv2d(no_in_f,2*no_in_f,1) # 1*1 conv
    self.relu = nn.ReLU()
  def forward(self, inp):
    x = self.relu(inp)
    x = self.conv1.forward(x)
    
    x = self.relu(x)
    x = self.masked.forward(x)
    
    x = self.relu(x)
    x = self.conv2.forward(x)

    x = x + inp
    return x
