

import os
import math
import numpy as np
from skimage.metrics import structural_similarity as SSIM
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import *
import logging
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import random
import sys
import torchvision.transforms as T
import pickle
from ddpm_downscale import *


def load_model(loading = "directly"):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default = 4)
    parser.add_argument('--dataset_size', type=int, required=False, default = 10000)
    parser.add_argument('--noise_schedule', type=str, required=False, default = "linear")
    parser.add_argument('--epochs', type=int, required=False, default = 500)
    parser.add_argument('--lr', type=float, required=False, default = 0.0)
    parser.add_argument('--dataset_type', type=str, required=False, default = "wind")
    parser.add_argument('--repeat_observations', type=int, required=False, default = 1)
    parser.add_argument('--cfg_proportion', type=float, required=False, default = 0)
    parser.add_argument('--image_size', type=int, required=False, default = None)
    parser.add_argument('--shuffle', type=bool, required=False, default = False)
    parser.add_argument('--resolution_ratio', type=int, required=False, default = 4)

    args = parser.parse_args()
    args.interp_mode = 'bicubic'
    args.proportion_train = 2.0
    args.lr = 3e-4 * 14 / args.batch_size

    args.dataset_path_hr = "/cluster/work/math/climate-downscaling/WiSoSuper_data/train/wind/middle_patch/HR"
    args.dataset_path_lr = "/cluster/work/math/climate-downscaling/WiSoSuper_data/train/wind/middle_patch/LR"


    args.c_in = 6
    args.c_out = 3
    args.image_size = 64
    device = 'cuda' # to do: change
    args.device = device
    args.n_example_imgs = 2000
    args.dataset_size == 10000
    args.noise_steps = 750

    model = UNet_downscale(c_in = args.c_in, c_out = args.c_out,
                                   img_size = args.image_size,
                                   interp_mode=args.interp_mode, device=device).to(device)
            #load the trained model
    model.load_state_dict(torch.load('models/DDPM_downscale_v2_fixeddata/ckpt.pt',
                                             map_location='cuda')) ########### need to change the path

    diffusion = Diffusion(img_size=args.image_size, device=device, \
                noise_steps=args.noise_steps, noise_schedule=args.noise_schedule)

    # OPTION DIRECTLY
    if loading == "directly":
        files = os.listdir(args.dataset_path_lr)
        first_file = files[0]

        path =  os.path.join(args.dataset_path_lr, first_file)
        images_lr = read_image(path, mode = ImageReadMode(3)).unsqueeze(0)
        for i in range(args.n_example_imgs):
            file = files[i + 1]
            path =  os.path.join(args.dataset_path_lr, file)
            img = read_image(path, mode = ImageReadMode(3)).unsqueeze(0)
            images_lr = torch.cat([images_lr, img], dim=0)

        norm = 255 / 2.0
        transform_lr = T.Compose([
            T.CenterCrop((args.image_size // args.resolution_ratio, args.image_size // args.resolution_ratio)),
            T.Normalize((norm, norm, norm), (norm, norm, norm))
            ])
        images_lr_save = images_lr
        images_lr = transform_lr(images_lr.float())
        print(images_lr_save.shape)
        print(images_lr.shape)

    if loading == "from_dataloader":
        if args.dataset_size == 10000:# fixed random permutation in this case
                # args.perm = np.arange(1, 10000)
                with open('data_permutation', 'rb') as data_permutation_file:
                    args.perm = pickle.load(data_permutation_file)
        dataloader, dataloader_test = get_data(args)
        it_test = iter(dataloader_test)
        for i in range(args.n_example_imgs//4):
            try:
                images_hr, images_lr = next(it_test)
            except StopIteration:
                iterloader = iter(dataloader_test)
                images_hr, images_lr = next(iterloader)
            images_lr = images_lr[0:4]
            images_hr = images_hr[0:4]
            images_hr_save =  (images_hr + 1) / 2.0 * 255
            images_hr_save = images_hr_save.type(torch.uint8)
            images_lr_save =  (images_lr + 1) / 2.0 * 255
            images_lr_save = images_lr_save.type(torch.uint8)

            print(images_lr_save.shape)
            print(images_lr.shape)
    #load

            sampled_images = diffusion.sample(model, n=len(images_lr), images_lr = images_lr, cfg_scale = 0) #cfg_scale
            for j in range(sampled_images.shape[0]):
                filename = os.path.join("Testing_v2_fixeddata/Image_test_generated", f"{4*i+j}_test_generated.jpg")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                save_images(sampled_images[j], filename)
                filename = os.path.join("Testing_v2_fixeddata/Image_test_lowers", f"{4*i+j}_test_lowres.jpg")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                save_images(images_lr_save[j], filename)
                filename = os.path.join("Testing_v2_fixeddata/Image_test_truth", f"{4*i+j}_test_truth.jpg")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                save_images(images_hr_save[j], filename)
    '''
    # OPTION DATALOADER
    if loading == "from_dataloader":
        if args.dataset_size == 10000:# fixed random permutation in this case
                with open('data_permutation', 'rb') as data_permutation_file:
                    args.perm = pickle.load(data_permutation_file)

            dataloader, dataloader_test = get_data(args)
            it_test = iter(dataloader_test)
            for i in range(args.n_example_imgs):
                images_hr, images_lr = next(it_test)
                images_lr = images_lr[0:4]
                images_hr = images_hr[0:4]
                images_hr_save =  (images_hr + 1) / 2.0 * 255
                images_hr_save = images_hr_save.type(torch.uint8)
                images_lr_save =  (images_lr + 1) / 2.0 * 255
                images_lr_save = images_lr_save.type(torch.uint8)

                print(images_lr_save.shape)
                print(images_lr.shape)



        #load
        model = UNet_downscale(c_in = args.c_in, c_out = args.c_out,
                               img_size = args.image_size,
                               interp_mode=args.interp_mode, device=device).to(device)
        #load the trained model
        model.load_state_dict(torch.load('.\ckpt.pt',
                                         map_location='cuda')) ########### need to change the path

        diffusion = Diffusion(img_size=args.image_size, device=device, \
            noise_steps=args.noise_steps, noise_schedule=args.noise_schedule)
        sampled_images = diffusion.sample(model, n=len(images_lr), images_lr = images_lr, cfg_scale = 0) #cfg_scale
        print(sampled_images.shape)
        for j in range(sampled_images.shape[0]):
            save_images(sampled_images[j], os.path.join("Image_test_generated", f"{4*i+j}_test_generated.jpg"))
            save_images(images_lr_save[j], os.path.join("Image_test_lowers", f"{4*i+j}_test_lowres.jpg"))
            save_images(images_hr_save[j], os.path.join("Image_test_truth", f"{4*i+j}_test_truth.jpg"))'''


if __name__ == '__main__':
    load_model(loading = "from_dataloader")
