

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
import torch.linalg as LA

def dist(x, y):
    # take matrix norm and sum across channel dimension
    # output will be a tensor of length x.shape[0]
    return torch.sum(LA.matrix_norm(x - y), dim = 1)

def energy_score(sample1, sample2, truth):
    return dist(sample1, truth) - 0.5 * dist(sample1, sample2)


def sample_multiple(loading = "directly"):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default = 1)
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
    args.lr = 3e-4 * 14 / args.batch_size

    args.dataset_path_hr = "/cluster/work/math/climate-downscaling/WiSoSuper_data/train/wind/middle_patch/HR"
    args.dataset_path_lr = "/cluster/work/math/climate-downscaling/WiSoSuper_data/train/wind/middle_patch/LR"

    # args.dataset_path_hr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch/HR"
    # args.dataset_path_lr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch/LR"

    args.c_in = 6
    args.c_out = 3
    args.image_size = 64
    # device = 'cpu' # to do: change
    device = 'cuda' # to do: change
    args.device = device
    args.n_example_imgs = 20
    args.n_rep_lowres = 100
    args.dataset_size == 10000
    args.noise_steps = 750

    model = UNet_downscale(c_in = args.c_in, c_out = args.c_out,
                                   img_size = args.image_size,
                                   interp_mode=args.interp_mode, device=device).to(device)
            #load the trained model
    model.load_state_dict(torch.load('models/DDPM_downscale_' + args.folder_name + '/ckpt.pt',
                                             map_location=device)) ########### need to change the path

    diffusion = Diffusion(img_size=args.image_size, device=device, \
                noise_steps=args.noise_steps, noise_schedule=args.noise_schedule)

    if loading == "from_dataloader":
        if args.dataset_size == 10000:# fixed random permutation in this case
                # args.perm = np.arange(1, 10000)
                with open('data_permutation', 'rb') as data_permutation_file:
                    args.perm = pickle.load(data_permutation_file)
        dataloader, dataloader_test = get_data(args)
        it_test = iter(dataloader_test)

        # scores = np.zeros(args.n_example_imgs)
        for i in range(args.n_example_imgs):
            try:
                images_hr, images_lr = next(it_test)
            except StopIteration:
                iterloader = iter(dataloader_test)
                images_hr, images_lr = next(iterloader)
            images_lr = images_lr
            images_hr = images_hr
            images_hr_save =  (images_hr + 1) / 2.0 * 255
            images_hr_save = images_hr_save.type(torch.uint8)
            images_lr_save =  (images_lr + 1) / 2.0 * 255
            images_lr_save = images_lr_save.type(torch.uint8)

            # bicubic interpolation
            images_bicubic = F.interpolate(images_lr.float(), size = [images_hr.shape[-1], images_hr.shape[-2]], mode = "bicubic")
            images_bicubic = images_bicubic.clamp(-1, 1)
            images_bicubic_save =  (images_bicubic + 1) / 2.0 * 255
            images_bicubic_save = images_bicubic_save.type(torch.uint8)

            images_lr = images_lr[0].repeat(5, 1, 1, 1)
            sampled_images = diffusion.sample(model, n=len(images_lr), images_lr = images_lr, cfg_scale = 0)
            # scores[i] = energy_score(sampled_images.float().to("cpu"),
            #    sampled_images2.float().to("cpu"),
            #    images_hr.float().to("cpu"))
            # filename = os.path.join("EnergyScoreTesting_" + args.folder_name + "/Scores", f"{i}_energy_score.pt")
            # os.makedirs(os.path.dirname(filename), exist_ok=True)
            # torch.save(scores, filename)

            filename = os.path.join("EnergyScoreTesting_" + args.folder_name + "/Image_test_generated", f"{i}_test_generated.jpg")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save_images(sampled_images, filename)
            filename = os.path.join("EnergyScoreTesting_" + args.folder_name + "/Image_test_lowers", f"{i}_test_lowres.jpg")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save_images(images_lr_save, filename)
            filename = os.path.join("EnergyScoreTesting_" + args.folder_name + "/Image_test_truth", f"{i}_test_truth.jpg")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save_images(images_hr_save, filename)
            filename = os.path.join("EnergyScoreTesting_" + args.folder_name + "/Image_test_bicubic", f"{i}_test_bicubic.jpg")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save_images(images_bicubic_save, filename)
            filename = os.path.join("EnergyScoreTesting_" + args.folder_name + "/Image_test_comparison", f"{i}_test_all.jpg")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            all = torch.cat([images_bicubic_save.to("cpu"), images_hr_save.to("cpu"), sampled_images.to("cpu")], dim = 0)
            save_images(all, filename)

if __name__ == '__main__':
    sample_multiple(loading = "from_dataloader")
