#!/user/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoFusion(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3,
            stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
            stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3,
            stride=1, padding=1)
        self.relu = nn.ReLU()

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)


    def forward(self, x):
        fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(fusecat)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)
        

        return ((fusecat * attn).sum(1)).unsqueeze(1)

