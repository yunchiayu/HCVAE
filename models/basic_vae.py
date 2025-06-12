import torch
import torch.nn as nn

import torch.nn.functional as F
from .basic_conv import BasicBlock, ResBasicBlock



# class Block(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#     def forward(self, x):
#         return x

class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv_query = nn.Conv2d(in_channels, in_channels, 1, padding=0, stride=1) # Hout = (Hin + 2 * pad -1 )+ 1
        self.conv_key   = nn.Conv2d(in_channels, in_channels, 1, padding=0, stride=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, 1, padding=0, stride=1)
        self.proj_out   = nn.Conv2d(in_channels, in_channels, 1, padding=0, stride=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.conv_query(x).view(B, C, H*W).contiguous()
        k = self.conv_key(x).view(B, C, H*W).contiguous()
        v = self.conv_value(x).view(B, C, H*W).contiguous()

        Q = q.permute((0, 2, 1)).contiguous() # (B, H*W, C)
        KT = k
        score = (Q @ KT) / torch.tensor(C).sqrt() # (B, H*W, C) @ (B, C, H*W) = (B, H*W, H*W)
        S = F.softmax(score, dim=-1) # (B, H*W, H*W)
        V = v.permute((0, 2, 1)).contiguous() # (B, H*W, C)

        SV = S @ V # (B, H*W, H*W) @ (B, H*W, C) = (B, H*W, C)
        SV = SV.permute((0, 2, 1)).contiguous() # (B, C, H*W)
        SV = SV.view(B, C, H, W).contiguous()   # (B, C, H, W)

        return x + self.proj_out(SV)


class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


class Encoder(nn.Module):
    def __init__(self, 
        in_channels = 3,
        ch=128,
        z_channels=64,
        ch_mult = (1, 2, 4),
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ch = ch
        self.z_channels = z_channels
        self.ch_mult = ch_mult
        self.num_resolutions = num_resolutions = len(ch_mult) - 1

        self.conv_in = torch.nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1) 

        self.downblocks = nn.ModuleList()
        for i_blk in range(num_resolutions):
            down = nn.Module()

            block_in = ch * ch_mult[i_blk]
            block_out = ch * ch_mult[i_blk+1]
            down.block = nn.Sequential(
                ResBasicBlock(block_in,  block_out),
                ResBasicBlock(block_out, block_out)
            )
            down.downsample = Downsample2x(block_out)

            self.downblocks.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResBasicBlock(block_out, block_out)
        self.mid.attn = Attention(block_out)
        self.mid.block_2 = ResBasicBlock(block_out, block_out)
        # conv_in:   (B,   3, 32, 32) -> (B, 128, 32, 32)
        # downscale: (B, 128, 32, 32) -> (B, 256, 16, 16) -> (B, 512,  8,  8)

        self.conv_out = torch.nn.Conv2d(block_out, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)

        # for i_blk in range(self.num_resolutions):
        #     h = self.downblocks[i_blk].block(h)
        #     h = self.downblocks[i_blk].downsample(h)
        for down in self.downblocks:
            h = down.block(h)
            h = down.downsample(h)
        
        h = self.mid.block_2(self.mid.attn(self.mid.block_1(h)))

        h = self.conv_out(h)

        return h
    
class Decoder(nn.Module):
    def __init__(self, 
        in_channels = 3,
        ch=128,
        z_channels=64,
        ch_mult = (1, 2, 4),
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ch = ch
        self.z_channels = z_channels
        self.ch_mult = ch_mult
        self.num_resolutions = num_resolutions = len(ch_mult) - 1

        block_in = ch * ch_mult[-1]
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResBasicBlock(block_in, block_in)
        self.mid.attn = Attention(block_in)
        self.mid.block_2 = ResBasicBlock(block_in, block_in)


        self.upblocks = nn.ModuleList()
        for i_blk in reversed(range(num_resolutions)):
            up = nn.Module()

            block_in = ch * ch_mult[i_blk+1]
            block_out = ch * ch_mult[i_blk]
            up.block = nn.Sequential(
                ResBasicBlock(block_in,  block_out),
                ResBasicBlock(block_out, block_out)
            )
            up.upsample = Upsample2x(block_out)

            self.upblocks.append(up)
        
        block_out = ch * ch_mult[0]
        self.conv_out = torch.nn.Conv2d(block_out, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        h = self.conv_in(x)

        h = self.mid.block_2(self.mid.attn(self.mid.block_1(h)))


        for up in self.upblocks:
            h = up.block(h)
            h = up.upsample(h)

        h = self.conv_out(h)

        return h