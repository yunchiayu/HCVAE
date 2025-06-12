import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResBasicBlock(nn.Module):
    def __init__(self, 
            in_channels:int, 
            out_channels:int, 
            kernel_size: tuple[int, int]=(3, 3), 
            stride: int = 1,
            padding: int = 1,
            **kwargs
        ):
        super().__init__()
        self.block1 = BasicBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # second convolution: in and out channels both equal out_channels
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2    = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.proj = nn.Conv2d(in_channels, out_channels, (1, 1) , stride=stride) if (stride != 1 or in_channels!=out_channels) else None

    def forward(self, x):
        identity = x

        out = self.block1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj is not None: 
            identity = self.proj(identity)

        out = self.relu(out + identity)

        return out
    

class MobileConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            **kwargs
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, drop_rate=0.0, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels = in_channels

        self.dwconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=3, 
            padding=1,
            **kwargs
        ) # -> (kernel_size, padding) = (7, 3) = (3, 1)

        self.ln = nn.LayerNorm(in_channels, eps=1e-6)

        # self.pw_conv1 = nn.Conv2d(in_channels,  out_channels*4, kernel_size=1)
        # self.gelu     = nn.GELU()
        # self.pw_conv2 = nn.Conv2d(out_channels*4, out_channels, kernel_size=1)

        self.pw_conv1 = nn.Linear(in_channels,  out_channels*4)
        self.gelu     = nn.GELU()
        self.pw_conv2 = nn.Linear(out_channels*4, out_channels)

        self.drop_path = nn.Dropout(drop_rate, inplace=False)

    def forward(self, x):
        identity = x
        out = self.dwconv(x) # -> (B, C, H, W)
        out = out.permute(0, 2, 3, 1) # -> (B, H, W, C)
        out = self.ln(out)
        out = self.pw_conv1(out) # -> (B, H, W, 4C)
        out = self.gelu(out)
        out = self.pw_conv2(out) # -> (B, H, W, C)
        out = out.permute(0, 3, 1, 2) # -> (B, C, H, W)

        out = out + self.drop_path(identity)

        return out

