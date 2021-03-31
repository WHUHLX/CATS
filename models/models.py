#!/user/bin/python
# -*- encoding: utf-8 -*-

import os, sys
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F
from os.path import isfile
from .cofusion import CoFusion


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.cfg = cfg

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,
                        stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                        stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                        stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)


        #lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1)
        self.conv1_2_down = nn.Conv2d(64, 21, 1)

        self.conv2_1_down = nn.Conv2d(128, 21, 1)
        self.conv2_2_down = nn.Conv2d(128, 21, 1)

        self.conv3_1_down = nn.Conv2d(256, 21, 1)
        self.conv3_2_down = nn.Conv2d(256, 21, 1)
        self.conv3_3_down = nn.Conv2d(256, 21, 1)

        self.conv4_1_down = nn.Conv2d(512, 21, 1)
        self.conv4_2_down = nn.Conv2d(512, 21, 1)
        self.conv4_3_down = nn.Conv2d(512, 21, 1)

        self.conv5_1_down = nn.Conv2d(512, 21, 1)
        self.conv5_2_down = nn.Conv2d(512, 21, 1)
        self.conv5_3_down = nn.Conv2d(512, 21, 1)

        #lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)

        self.weight_deconv2 =  make_bilinear_weights(4, 1).cuda()
        self.weight_deconv3 =  make_bilinear_weights(8, 1).cuda()
        self.weight_deconv4 =  make_bilinear_weights(16, 1).cuda()
        self.weight_deconv5 =  make_bilinear_weights(16, 1).cuda()


        self.attention = CoFusion(5, 5)

    def init_weight(self):

        print("=> Initialization by Gaussian(0, 0.01)")

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                ly.weight.data.normal_(0, 0.01)
                if not ly.bias is None: ly.bias.data.zero_()

        if self.cfg.pretrained:
            if not isfile(self.cfg.pretrained):
                print("No pretrained VGG16 model found at '{}'".format(self.cfg.pretrained))

            else:
                print("=> Initialize VGG16 backbone")

                state_dict = torch.load(self.cfg.pretrained, map_location=torch.device("cpu"))

                self_state_dict = self.state_dict()
                for k, v in self_state_dict.items():
                    if k in state_dict.keys():
                        self_state_dict.update({k: state_dict[k]})
                        print("*** Load {} ***".format(k))

                self.load_state_dict(self_state_dict)
                print("=> Pretrained Loaded")

    def load_checkpoint(self):
        if isfile(self.cfg.resume):
            print("=> Loading checkpoint '{}'".format(self.cfg.resume))
            checkpoint = torch.load(self.cfg.resume)
            self.load_state_dict(checkpoint['state_dict'])
            print("=> Loaded checkpoint '{}'"
                  .format(self.cfg.resume))

        else:
            print("=> No checkpoint found at '{}'".format(self.cfg.resume))

    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1   = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2   = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3   = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4   = self.maxpool4(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        ## transpose and crop way


        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, self.weight_deconv5, stride=8)


        ### center crop
        so1 = crop_bdcn(so1_out, img_H, img_W, 0 , 0)
        so2 = crop_bdcn(upsample2, img_H, img_W , 1, 1)
        so3 = crop_bdcn(upsample3, img_H, img_W , 2, 2)
        so4 = crop_bdcn(upsample4, img_H, img_W , 4, 4)
        so5 = crop_bdcn(upsample5, img_H, img_W , 0, 0)

        results = [so1, so2, so3, so4, so5]
        fuse = self.attention(results)

        results.append(fuse)


        results = [torch.sigmoid(r) for r in results]
        return results



# Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
def crop_bdcn(data1, h, w , crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert(h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h+h, crop_w:crop_w+w]
    return data

def crop(variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return variable[:, :, y1 : y1 + th, x1 : x1 + tw]

# make a bilinear interpolation kernel
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w

def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)
