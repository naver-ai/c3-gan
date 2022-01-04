import torch
import torch.nn as nn

from base_layer import *
from config import cfg



class bg_generator(nn.Module):
    def __init__(self, ngf=512):
        super(bg_generator, self).__init__()
        self.z_dim = cfg.GAN.Z_DIM
        self.ngf = ngf
        self.fc = nn.Sequential(nn.Linear(self.z_dim, ngf*4*4*2, bias=False), nn.BatchNorm1d(ngf*4*4*2), GLU())
        self.layers = nn.Sequential(upBlock(ngf, ngf//2),
                                    upBlock(ngf//2, ngf//4),
                                    upBlock(ngf//4, ngf//8),
                                    upBlock(ngf//8, ngf//32),
                                    upBlock(ngf//32, ngf//32),
                                    multi_ResBlock(3, ngf//32),
                                    nn.Conv2d(ngf//32, 3, 3, 1, 1, bias=False),
                                    nn.Tanh())

    def forward(self, z):
        out = self.fc(z).view(-1, self.ngf, 4, 4) 
        out = self.layers(out) 
        return out


class fg_generator(nn.Module):
    def __init__(self, c_dim, ngf=512):
        super(fg_generator, self).__init__()
        self.z_dim = cfg.GAN.Z_DIM 
        self.cz_dim = cfg.GAN.CZ_DIM 
        self.c_dim = c_dim
        self.ngf = ngf
        self.fc = nn.Sequential(nn.Linear(self.z_dim, ngf*4*4*2, bias=False), nn.BatchNorm1d(ngf*4*4*2), GLU())
        self.emb_c = nn.Sequential(nn.Linear(self.c_dim, self.cz_dim*2*2), nn.BatchNorm1d(self.cz_dim*2*2), GLU())
        self.base = nn.Sequential(upBlock(ngf + self.cz_dim, ngf//2),
                                    upBlock(ngf//2, ngf//4),
                                    upBlock(ngf//4, ngf//8),
                                    upBlock(ngf//8, ngf//32),
                                    upBlock(ngf//32, ngf//32),
                                    multi_ResBlock(3, ngf//32))
        self.to_mask = nn.Sequential(sameBlock(ngf//32, ngf//32),
                                     nn.Conv2d(ngf//32, 1, 3, 1, 1, bias=False))
        self.to_img = nn.Sequential(sameBlock(self.c_dim+ngf//32, ngf//32),
                                    multi_ResBlock(2, ngf//32),
                                    sameBlock(ngf//32, ngf//32),
                                    nn.Conv2d(ngf//32, 3, 3, 1, 1, bias=False), nn.Tanh())

    def forward(self, z, c, cz):
        ## get c' from c
        c_ = self.emb_c(c)
        c_mu = c_[:, :self.cz_dim]
        c_std = c_[:, self.cz_dim:]
        cz_ = c_mu + c_std.exp()*cz
        cz_ = cz_.view(-1, self.cz_dim, 1, 1).repeat(1, 1, 4, 4)
        ## get base_feat
        out = self.fc(z).view(-1, self.ngf, 4, 4) 
        out = self.base(torch.cat((out, cz_), 1))
        ## get fg_mask
        out_mask = torch.sigmoid(self.to_mask(out))
        ## get fg_image
        h, w  = out.size(2),out.size(3)
        c = c.view(-1, self.c_dim, 1, 1).repeat(1, 1, h, w) 
        out = torch.cat((out, c), 1)
        out_img = self.to_img(out)
        return out_mask, out_img


class Generator(nn.Module):
    def __init__(self, c_dim):
        super(Generator, self).__init__()
        self.bg_gen = bg_generator()
        self.fg_gen = fg_generator(c_dim)

    def forward(self, z, cz, c, grid=None):
        bg_img = self.bg_gen(z) # get background image
        fg_mask, fg_img = self.fg_gen(z, c, cz) # get foreground image
        if grid != None:
            fg_mask = F.grid_sample(fg_mask, grid, align_corners=True)
            fg_img = F.grid_sample(fg_img, grid, align_corners=True)
        final_img = bg_img*(1-fg_mask) + fg_img*fg_mask 
        return bg_img, fg_mask, fg_img, final_img


class Discriminator(nn.Module):
    def __init__(self, c_dim, ndf=64):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.c_dim = c_dim
        self.base = encode_img()
        self.info_head = nn.Sequential(nn.Conv2d(ndf*8, ndf*8, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(ndf*8),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=4))
        self.rf_head = nn.Sequential(nn.Conv2d(ndf*8, 1, kernel_size=4, stride=4))
        self.centroids = nn.Linear(self.c_dim, ndf*8)

    def forward(self, x, eye, masked_x=None):
        out = self.base(x)
        info = self.info_head(out).view(-1, self.ndf*8)
        rf = self.rf_head(out).view(-1, 1)
        class_emb = self.centroids(eye)
        if masked_x != None:
            fg_out = self.base(masked_x)
            fg_info = self.info_head(fg_out).view(-1, self.ndf*8)
            return info, rf, class_emb, fg_info
        else:
            return info, rf, class_emb

