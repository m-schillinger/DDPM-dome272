#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:50:57 2022

@author: mschillinger
"""

from utils import *
from ddpm_downscale_euler_run2_bs15 import Diffusion

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.dataset_path_hr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch_subset/HR"
args.dataset_path_lr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch_subset/LR"
args.dataset_type = "wind"
args.dataset_size = 100
args.batch_size = 1
args.device = "cpu"
dataloader = get_data(args)
dm = Diffusion(device = "cpu")

it = iter(dataloader)

for i in range(10):
    train_hr, train_lr = next(it)
    print(f"Train hr shape: {train_hr.size()}")
    print(f"Train lr shape: {train_lr.size()}")
    img = train_hr[0].unsqueeze(0).float()
    lr = train_lr[0].unsqueeze(0).float()
    y = F.interpolate(lr, size = [64,64], mode = 'bilinear')
    # y = F.interpolate(lr, size = [64, 64], mode = 'bicubic')
    img = img.squeeze().permute(1, 2, 0).byte()
    lr = lr.squeeze().permute(1, 2, 0).byte()
    y = y.squeeze().permute(1,2,0).byte()
    plt.imshow(lr)
    plt.show()
    plt.imshow(img)
    plt.show()
    #plt.imshow(y)
    #plt.show()
    
    noisy = dm.noise_images(train_hr, 100)
    z = noisy[0].unsqueeze(0).float()
    z = z.squeeze().permute(1, 2, 0).byte()
    plt.imshow(z)
    plt.show()
