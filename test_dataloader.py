#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 18:49:06 2023

@author: mschillinger
"""

from utils import *

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default = 5)
    parser.add_argument('--dataset_size', type=int, required=False, default = 200)
    parser.add_argument('--noise_schedule', type=str, required=False, default = "linear")
    parser.add_argument('--epochs', type=int, required=False, default = 500)
    parser.add_argument('--lr', type=float, required=False, default = 0.0)
    parser.add_argument('--dataset_type', type=str, required=False, default = "wind")
    parser.add_argument('--repeat_observations', type=int, required=False, default = 1)
    parser.add_argument('--cfg_proportion', type=float, required=False, default = 0)
    parser.add_argument('--image_size', type=int, required=False, default = 64)
    parser.add_argument('--shuffle', type=bool, required=False, default = False)
    args = parser.parse_args()

    args.dataset_path_hr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch/HR"
    args.dataset_path_lr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch/LR"
    args.n_example_imgs = 5
    
    dataloader, dataloader_test = get_data(args)
    it = iter(dataloader)
    for i in range(5):
        train_hr, train_lr = next(it)
        print(f"Train hr shape: {train_hr.size()}")
        print(f"Train lr shape: {train_lr.size()}")
        train_hr = (train_hr + 1) / 2.0 * 255
        train_lr = (train_lr + 1) / 2.0 * 255

        img = train_hr[0].unsqueeze(0).float()
        lr = train_lr[0].unsqueeze(0).float()
        y = F.interpolate(lr, size = [64,64], mode = 'bilinear')
        img = img.squeeze().permute(1, 2, 0).byte()
        lr = lr.squeeze().permute(1, 2, 0).byte()
        y = y.squeeze().permute(1,2,0).byte()
        plt.imshow(lr)
        plt.show()
        plt.imshow(img)
        plt.show()
        plt.imshow(y)
        plt.show()
        
        save_images(train_lr.type(torch.uint8), path = "test_dataload_lr.png")
        save_images(train_hr.type(torch.uint8), path = "test_dataload_hr.png")
        
    it2 = iter(dataloader_test)
    for i in range(5):
        train_hr, train_lr = next(it2)
        print(f"Train hr shape: {train_hr.size()}")
        print(f"Train lr shape: {train_lr.size()}")
        train_hr = (train_hr + 1) / 2.0 * 255
        train_lr = (train_lr + 1) / 2.0 * 255

        img = train_hr[0].unsqueeze(0).float()
        lr = train_lr[0].unsqueeze(0).float()
        y = F.interpolate(lr, size = [64,64], mode = 'bilinear')
        img = img.squeeze().permute(1, 2, 0).byte()
        lr = lr.squeeze().permute(1, 2, 0).byte()
        y = y.squeeze().permute(1,2,0).byte()
        plt.imshow(lr)
        plt.show()
        plt.imshow(img)
        plt.show()
        plt.imshow(y)
        plt.show()
        
        save_images(train_lr.type(torch.uint8), path = "test_dataload_lr2.png")
        save_images(train_hr.type(torch.uint8), path = "test_dataload_hr2.png")