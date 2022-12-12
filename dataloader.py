#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:07:48 2022

@author: mschillinger
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io.image import ImageReadMode
from torchvision.io import read_image

class DownscalingDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, 
                 transform_hr=None, transform_lr=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        self.max_t = len(os.listdir(self.hr_dir)) * 2

    def __len__(self):
        return len(os.listdir(self.hr_dir))

    def __getitem__(self, idx):
        if(idx * 4 > self.max_t):
            prefix = "va"
            t = (idx - 1)*4 - self.max_t
            filename = "{}_{}.png".format(prefix, t)
        else:
            prefix = "ua"
            t = idx * 4
            filename = "{}_{}.png".format(prefix, t)
        hr_path = os.path.join(self.hr_dir, filename)
        lr_path = os.path.join(self.lr_dir, filename)
        # mode ensures RGB values (i.e. only three channels, no alpha channel)
        image_hr = read_image(hr_path, mode = ImageReadMode(3))
        print(image_hr.shape)
        image_lr = read_image(lr_path, mode = ImageReadMode(3))
        if self.transform_hr:
            image_hr = self.transform_hr(image_hr)
        if self.transform_lr:
            image_lr = self.transform_hr(image_lr)
        return image_hr, image_lr

if __name__ == '__main__':
    dataset = DownscalingDataset("/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch/HR", 
                                 "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch/LR")
    
    dataloader = DataLoader(dataset)
    it = iter(dataloader)
    for i in range(10):
        train_hr, train_lr = next(it)
        print(f"Train hr shape: {train_hr.size()}")
        print(f"Train lr shape: {train_lr.size()}")
        img = train_hr[0].squeeze()
        img = img.permute(1, 2, 0)
        lr = train_lr[0].squeeze()
        lr = lr.permute(1, 2, 0)
        plt.imshow(img)
        plt.show()
        plt.imshow(lr)
        plt.show()