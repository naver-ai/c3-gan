import math

import torch
import torch.nn as nn
import torch.optim as optim

from model import Generator, Discriminator



## for model initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def define_optimizers(netG, netD):
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return optimizerG, optimizerD

def load_network(c_dim):
    # G
    netG = Generator(c_dim)
    netG.apply(weights_init)
    netG = nn.DataParallel(netG)
    netG = netG.cuda() 
    # D
    netD = Discriminator(c_dim)
    netD.apply(weights_init)
    netD = nn.DataParallel(netD)
    netD = netD.cuda()
    return netG, netD


## loss implementation
def binary_entropy(p):
    return -p*torch.log2(p+1e-6) - (1-p)*torch.log2(1-p+1e-6)


## misc
def postprocess(x):
    return x.add(1).div(2).clamp(0, 1)

def to_rad(deg):
    return deg/180*math.pi
