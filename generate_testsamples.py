import os
import math
import numpy as np
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
import torch.nn.functional as F
import pickle
from ddpm_downscale import *


def load_model():
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
    parser.add_argument('--folder_name', type=str, required=False, default = "v2_fixeddata")

    args = parser.parse_args()
    args.interp_mode = 'bicubic'
    args.proportion_train = 2.0
    args.lr = 0.0001

    args.dataset_path_hr = "/cluster/work/math/climate-downscaling/WiSoSuper_data/train/wind/middle_patch/HR"
    args.dataset_path_lr = "/cluster/work/math/climate-downscaling/WiSoSuper_data/train/wind/middle_patch/LR"
    # args.dataset_path_hr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch/HR"
    # args.dataset_path_lr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch/LR"

    args.c_in = 6
    args.c_out = 3
    args.image_size = 64
    device = 'cuda'
    args.device = device
    args.n_example_imgs = 2000
    args.dataset_size == 10000
    args.noise_steps = 750

    model = UNet_downscale(c_in = args.c_in, c_out = args.c_out,
                                   img_size = args.image_size,
                                   interp_mode=args.interp_mode, device=device).to(device)
    # load the trained model
    model.load_state_dict(torch.load('models/DDPM_downscale_' + args.folder_name + '/ckpt.pt',
                                             map_location=device))

    # set up diffusion model
    diffusion = Diffusion(img_size=args.image_size, device=device, \
                noise_steps=args.noise_steps, noise_schedule=args.noise_schedule)

    if args.dataset_size == 10000:
            # fixed random permutation in this case
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

        # bicubic interpolation for comparison
        images_bicubic = F.interpolate(images_lr.float(), size = [images_hr.shape[-1], images_hr.shape[-2]], mode = "bicubic")
        images_bicubic = images_bicubic.clamp(-1, 1)
        images_bicubic_save =  (images_bicubic + 1) / 2.0 * 255
        images_bicubic_save = images_bicubic_save.type(torch.uint8)

        # sample new images
        sampled_images = diffusion.sample(model, n=len(images_lr), images_lr = images_lr, cfg_scale = 0)

        # save everything
        for j in range(sampled_images.shape[0]):
            filename = os.path.join("Testing_" + args.folder_name + "/Image_test_generated", f"{4*i+j}_test_generated.jpg")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save_images(sampled_images[j], filename)
            filename = os.path.join("Testing_" + args.folder_name + "/Image_test_lowers", f"{4*i+j}_test_lowres.jpg")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save_images(images_lr_save[j], filename)
            filename = os.path.join("Testing_" + args.folder_name + "/Image_test_truth", f"{4*i+j}_test_truth.jpg")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save_images(images_hr_save[j], filename)
            filename = os.path.join("Testing_" + args.folder_name + "/Image_test_bicubic", f"{4*i+j}_test_bicubic.jpg")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save_images(images_bicubic_save[j], filename)

if __name__ == '__main__':
    load_model()
