#!/user/bin/python
# -*- encoding: utf-8 -*-

from torch.utils import data
import os
from os.path import join, basename
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
from .transform import *

def prepare_image_PIL(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im


class MyDataLoader(data.Dataset):
    """
    Dataloader
    """
    def __init__(self, root='./Data/NYUD', split='train', transform=True):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'train.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

        # pre-processing
        if self.transform:
            self.trans = Compose([
                ColorJitter(
                    brightness = 0.5,
                    contrast = 0.5,
                    saturation = 0.5),
                #RandomCrop((512, 512))
                ])

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()

            label = Image.open(join(self.root, lb_file))
            img = Image.open(join(self.root, img_file))

            if self.transform:
                im_lb = dict(im = img, lb = label)
                im_lb = self.trans(im_lb)
                img, label = im_lb['im'], im_lb['lb']

            img = np.array(img, dtype=np.float32)
            img = prepare_image_PIL(img)

            label = np.array(label, dtype=np.float32)

            if label.ndim == 3:
                label = np.squeeze(label[:, :, 0])
            assert label.ndim == 2

            label = label[np.newaxis, :, :]
            label[label == 0] = 0
            label[np.logical_and(label>0, label<=100)] = 2
            label[label > 100] = 1

            return img, label, basename(img_file).split('.')[0]
        else:
            img_file = self.filelist[index].rstrip()
            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_PIL(img)
            return img, basename(img_file).split('.')[0]
