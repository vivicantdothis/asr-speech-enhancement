"""
model.py is a research grade U-Net for speech enhancement where the input is
(B,1,F,T) where B=batch size, F= frequency bins, and T=time frames

the output is an estimated IDEAL RATIO MASK (IRM)
(B,1,F,T).
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .config import (INPUT_CHANNELS,OUTPUT_CHANNELS,ENCODER_CHANNELS,BOTTLENECK_CHANNELS,DROPOUT,USE_BATCH_NORM,)
from .blocks import (DoubleConv,DownBlock,UpBlock,OutputBlock,)

class UNetSpeechEnhancer(nn.Module):
    #input is a log magnitude spectrogram and the output is an ideal ratio mask
    def __init__(self):
        super().__init__()
        #ENCODER
        self.input_block=DoubleConv(INPUT_CHANNELS,ENCODER_CHANNELS[0],batch_norm=USE_BATCH_NORM,)
        self.down1=DownBlock(ENCODER_CHANNELS[0],ENCODER_CHANNELS[1],batch_norm=USE_BATCH_NORM,)
        self.down2=DownBlock(ENCODER_CHANNELS[1],ENCODER_CHANNELS[2],batch_norm=USE_BATCH_NORM,)
        self.down3=DownBlock(ENCODER_CHANNELS[2],ENCODER_CHANNELS[3],batch_norm=USE_BATCH_NORM,dropout=DROPOUT,)

        #BOTTLENECK
        self.bottleneck=DownBlock(ENCODER_CHANNELS[3],BOTTLENECK_CHANNELS,batch_norm=USE_BATCH_NORM,dropout=DROPOUT,)
        #DECODER
        self.up1=UpBlock(BOTTLENECK_CHANNELS,ENCODER_CHANNELS[3],ENCODER_CHANNELS[3],batch_norm=USE_BATCH_NORM,dropout=DROPOUT,)
        self.up2=UpBlock(ENCODER_CHANNELS[3],ENCODER_CHANNELS[2],ENCODER_CHANNELS[2],batch_norm=USE_BATCH_NORM,)
        self.up3=UpBlock(ENCODER_CHANNELS[2],ENCODER_CHANNELS[1],ENCODER_CHANNELS[1],batch_norm=USE_BATCH_NORM,)
        self.up4=UpBlock(ENCODER_CHANNELS[1],ENCODER_CHANNELS[0],ENCODER_CHANNELS[0],batch_norm=USE_BATCH_NORM,)

        #OUTPUT
        self.output=OutputBlock(ENCODER_CHANNELS[0],OUTPUT_CHANNELS,)
    
    #FORWARD
    def forward(self,x):
        x1=self.input_block(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)

        #BOTTLENECK
        b=self.bottleneck(x4)
        #DECODER
        d1=self.up1(b,x4)
        d2=self.up2(d1,x3)
        d3=self.up3(d2,x2)
        d4=self.up4(d3,x1)
        #OUTPUT MASK
        mask=self.output(d4)
        return mask
    
def build_model():
    return UNetSpeechEnhancer()