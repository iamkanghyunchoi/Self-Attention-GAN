import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np
from g_mlp_pytorch import Residual, PreNorm, gMLPBlock
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, attn_arch="sagan"):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn_arch = attn_arch

        if attn_arch == "sagan":
            self.attn1 = Self_Attn( 128, 'relu') # 32
            self.attn2 = Self_Attn( 64,  'relu') # 64
        elif attn_arch == "gmlp":
            self.to_embed1 = nn.Sequential(
                Rearrange('b c h w -> b (h w) c'),
                nn.Linear(128, 128)
            )
            self.to_embed2 = nn.Sequential(
                Rearrange('b c h w -> b (h w) c'),
                nn.Linear(64, 64)
            )

            self.attn1 = gMLPBlock(dim=128, heads=1, dim_ff = 128*4, seq_len = 16**2, attn_dim = None)
            self.attn2 = gMLPBlock(dim=64, heads=1, dim_ff = 64*4, seq_len = 32**2, attn_dim = None)

            self.to_img1 = nn.Sequential(
                nn.Linear(128, 128),
                Rearrange('b (h w) c -> b c h w', h=16, w=16)
            )
            self.to_img2 = nn.Sequential(
                nn.Linear(64, 64),
                Rearrange('b (h w) c -> b c h w', h=32, w=32)
            )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z) # 512, 4x4
        out=self.l2(out) # 256, 8x8

        out=self.l3(out) # 128, 16x16
        if self.attn_arch == "sagan":
            out,p1 = self.attn1(out)
        elif self.attn_arch == "gmlp":
            out = self.to_embed1(out)
            out = self.attn1(out)
            out = self.to_img1(out)
            p1 = None

        out=self.l4(out) # 64, 32x32
        if self.attn_arch == "sagan":
            out,p2 = self.attn2(out)
        elif self.attn_arch == "gmlp":
            out = self.to_embed2(out)
            out = self.attn2(out)
            out = self.to_img2(out)
            p2 = None
        out=self.last(out)

        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64, attn_arch="sagan"):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn_arch = attn_arch

        if attn_arch == "sagan":
            self.attn1 = Self_Attn( 256, 'relu')
            self.attn2 = Self_Attn( 512,  'relu')
        elif attn_arch == "gmlp":
            self.to_embed1 = nn.Sequential(
                Rearrange('b c h w -> b (h w) c'),
                nn.Linear(256, 256)
            )
            self.to_embed2 = nn.Sequential(
                Rearrange('b c h w -> b (h w) c'),
                nn.Linear(512, 512)
            )
            self.attn1 = gMLPBlock(dim=256, heads=1, dim_ff=256*4, seq_len=8**2, attn_dim=None)
            self.attn2 = gMLPBlock(dim=512, heads=1, dim_ff=512*4, seq_len=4**2, attn_dim=None)

            self.to_img1 = nn.Sequential(
                nn.Linear(256, 256),
                Rearrange('b (h w) c -> b c h w', h=8, w=8)
            )
            self.to_img2 = nn.Sequential(
                nn.Linear(512, 512),
                Rearrange('b (h w) c -> b c h w', h=4, w=4)
            )

    def forward(self, x):
        out = self.l1(x) # 64, 32x32
        out = self.l2(out) # 128, 16x16
        out = self.l3(out) # 256, 8x8
        if self.attn_arch == "sagan":
            out,p1 = self.attn1(out)
        elif self.attn_arch == "gmlp":
            out = self.to_embed1(out)
            out = self.attn1(out)
            out = self.to_img1(out)
            p1 = None

        out=self.l4(out) # 512, 4x4
        if self.attn_arch == "sagan":
            out,p2 = self.attn2(out)
        elif self.attn_arch == "gmlp":
            out = self.to_embed2(out)
            out = self.attn2(out)
            out = self.to_img2(out)
            p2 = None

        out=self.last(out) # 1

        return out.squeeze(), p1, p2
