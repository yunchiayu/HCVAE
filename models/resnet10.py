import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_conv import BasicBlock, ResBasicBlock

class ResNet10(nn.Module):
    
    def __init__(self, in_channel:int=3, num_classes:int=100, use_bn: bool = True):
        super(ResNet10, self).__init__()

        channel_1 = 64
        channel_2 = 64
        channel_3 = 128
        channel_4 = 256
        channel_5 = 512

        self.conv1 = BasicBlock(in_channel,  channel_1, kernel_size=(7,7), stride=2, padding=3)
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d((3, 3), stride=2, padding=1), # (16, 16) -> (8, 8)
            ResBasicBlock(channel_1, channel_2, kernel_size=(3,3), stride=1, padding=1)   
        )
        self.conv3_x = ResBasicBlock(channel_2, channel_3, kernel_size=(3,3), stride=1, padding=1) 
        self.conv4_x = ResBasicBlock(channel_3, channel_4, kernel_size=(3,3), stride=1, padding=1) 
        self.conv5_x = ResBasicBlock(channel_4, channel_5, kernel_size=(3,3), stride=2, padding=1) 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # (512, 4, 4) -> (512, 1, 1)
        self.classifier = nn.Linear(channel_5, num_classes)

  
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x) # -> 
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x) # -> (B, 512, 4, 4)

        x = self.avgpool(x) # -> (B, 512, 1, 1)
        x = x.squeeze() # -> (B, 512)

        score = self.classifier(x) # -> (B. num_classes)

        return score


