"""
this file contains reusable building blocks for the U-Net speech enhancement model.
it contains DoubleConv, DownBlock, UpBlock, OutputBlock, which are all modules reused
throughout the CNN architecture."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

#DOUBLE CONVOLUTION BLOCK
class DoubleConv(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,batch_norm:bool=True,dropout:float=0.0,):
        super().__init__()
        layers=[]
        #first convolution
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=not batch_norm,))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout>0:
            layers.append(nn.Dropout2d(dropout))
        #second convolution
        layers.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=not batch_norm,))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block=nn.Sequential(*layers)
    def forward(self,x):
        return self.block(x)
    
#DOWNSAMPLING BLOCK
class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,batch_norm=True,dropout=0.0,):
        super().__init__()
        self.pool=nn.MaxPool2d(2)
        self.conv=DoubleConv(in_channels,out_channels,batch_norm=batch_norm,dropout=dropout,)
    def forward(self,x):
        x=self.pool(x)
        x=self.conv(x)
        return x

#UPSAMPLING BLOCK
class UpBlock(nn.Module):
    def __init__(self,in_channels,skip_channels,out_channels,batch_norm=True,dropout=0.0,):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2,)
        self.conv=DoubleConv(out_channels+skip_channels,out_channels,batch_norm=batch_norm,dropout=dropout,)
    def forward(self,x,skip):
        x=self.up(x)
        #handle odd input dimensions
        if x.shape[-2:]!=skip.shape[-2:]:
            x=F.interpolate(x,size=skip.shape[-2:],mode="bilinear",align_corners=False,)
        x=torch.cat([skip,x],dim=1)
        x=self.conv(x)
        return x

class OutputBlock(nn.Module):
    def __init__(self,in_channels,out_channels=1,):
        super().__init__()
        self.output=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1,),nn.Sigmoid(),)
    def forward(self,x):
        return self.output(x)
    
        
        
