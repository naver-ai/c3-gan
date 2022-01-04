import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Upsample



class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def upBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                          nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                          nn.BatchNorm2d(out_planes*2),
                          GLU())
    return block


def sameBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                          nn.BatchNorm2d(out_planes*2),
                          GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel_num, channel_num*2, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(channel_num*2),
                                   GLU(),
                                   nn.Conv2d(channel_num, channel_num, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(channel_num))

    def forward(self, x):
        return x + self.block(x)


def multi_ResBlock(num_residual, ngf):
    layers = []
    for _ in range(num_residual):
        layers.append(ResBlock(ngf))
    return nn.Sequential(*layers)


def encode_img(ndf=64, in_c=3):
    layers = nn.Sequential(
        nn.Conv2d(in_c, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return layers
