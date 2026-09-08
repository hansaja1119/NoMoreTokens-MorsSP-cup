"""Compact residual restoration network using the published NAF block design.

Architecture reference: Chen et al., Simple Baselines for Image Restoration,
ECCV 2022, https://github.com/megvii-research/NAFNet . This implementation
uses ordinary PyTorch operators and no compiled extensions or external weights.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class ChannelNorm(nn.Module):
    def __init__(self,c):
        super().__init__();self.weight=nn.Parameter(torch.ones(1,c,1,1));self.bias=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        # Accumulate variance in float32 for stable mixed-precision training.
        xf=x.float();mean=xf.mean(1,keepdim=True);var=(xf-mean).square().mean(1,keepdim=True)
        return ((xf-mean)*torch.rsqrt(var+1e-6)*self.weight+self.bias).to(x.dtype)

class Gate(nn.Module):
    def forward(self,x):
        a,b=x.chunk(2,dim=1);return a*b

class NAFBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.norm1=ChannelNorm(c);self.norm2=ChannelNorm(c)
        self.expand=nn.Conv2d(c,2*c,1);self.depth=nn.Conv2d(2*c,2*c,3,padding=1,groups=2*c)
        self.gate=Gate();self.attention=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,c,1))
        self.project=nn.Conv2d(c,c,1)
        self.ffn=nn.Sequential(nn.Conv2d(c,2*c,1),Gate(),nn.Conv2d(c,c,1))
        self.beta=nn.Parameter(torch.zeros(1,c,1,1));self.gamma=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        z=self.gate(self.depth(self.expand(self.norm1(x))))
        x=x+self.beta*self.project(z*self.attention(z))
        return x+self.gamma*self.ffn(self.norm2(x))

class Denoiser(nn.Module):
    def __init__(self,width=16,hybrid=True):
        super().__init__();self.config=dict(width=width,hybrid=hybrid)
        self.intro=nn.Conv2d(9 if hybrid else 3,width,3,padding=1)
        self.encoders=nn.ModuleList();self.downs=nn.ModuleList()
        c=width
        for blocks in [1,1,2,4]:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(blocks)]))
            self.downs.append(nn.Conv2d(c,2*c,2,stride=2));c*=2
        self.middle=nn.Sequential(*[NAFBlock(c) for _ in range(4)])
        self.ups=nn.ModuleList();self.decoders=nn.ModuleList()
        for _ in range(4):
            self.ups.append(nn.Sequential(nn.Conv2d(c,2*c,1,bias=False),nn.PixelShuffle(2)));c//=2
            self.decoders.append(NAFBlock(c))
        self.ending=nn.Conv2d(width,3,3,padding=1)
        nn.init.zeros_(self.ending.weight);nn.init.zeros_(self.ending.bias)
    def forward(self,inputs):
        h,w=inputs.shape[-2:];ph=(-h)%16;pw=(-w)%16
        padded=F.pad(inputs,(0,pw,0,ph),mode='replicate') if ph or pw else inputs
        anchor=padded[:,3:6] if self.config['hybrid'] else padded[:,:3]
        x=self.intro(padded);skips=[]
        for encoder,down in zip(self.encoders,self.downs):
            x=encoder(x);skips.append(x);x=down(x)
        x=self.middle(x)
        for up,decoder,skip in zip(self.ups,self.decoders,reversed(skips)):
            x=decoder(up(x)+skip)
        return (anchor+self.ending(x))[...,:h,:w]
