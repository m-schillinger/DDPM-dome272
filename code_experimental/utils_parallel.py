import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
from torchvision.io.image import ImageReadMode
from torchvision.io import read_image
import torch.nn.functional as F

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


class DownscalingDataset(Dataset):
    def __init__(self, hr_dir, lr_dir,
                 transform_hr=None, transform_lr=None, max_len=1e6):
        print(hr_dir)
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        self.max_t = (len(os.listdir(self.hr_dir)) - 2) * 2
        self.max_len = max_len

    def __len__(self):
        return np.min([len(os.listdir(self.hr_dir)), self.max_len])

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
        image_hr = read_image(hr_path, mode = ImageReadMode(3)).float()
        image_lr = read_image(lr_path, mode = ImageReadMode(3)).float()
        if self.transform_hr:
            image_hr = self.transform_hr(image_hr)
        if self.transform_lr:
            image_lr = self.transform_hr(image_lr)
        return image_hr, image_lr

class DownscalingTemperatureDataset(Dataset):
    def __init__(self, hr_dir, lr_dir,
                 transform_hr=None, transform_lr=None, max_len = 1e6):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        self.max_len = max_len

    def __len__(self):
        return np.min([len(os.listdir(self.hr_dir)), self.max_len])


    def __getitem__(self, idx):
        filename_hr = "tas_daily_highres_hist-scen_{}.png".format(idx)
        filename_lr = "tas_daily_lowres_hist-scen_{}.png".format(idx)
        hr_path = os.path.join(self.hr_dir, filename_hr)
        lr_path = os.path.join(self.lr_dir, filename_lr)
        # mode ensures RGB values (i.e. only three channels, no alpha channel)
        image_hr = read_image(hr_path, mode = ImageReadMode(3)).float()
        image_lr = read_image(lr_path, mode = ImageReadMode(3)).float()
        if self.transform_hr:
            image_hr = self.transform_hr(image_hr)
        if self.transform_lr:
            image_lr = self.transform_hr(image_lr)
        return image_hr, image_lr


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    if args.dataset_type == "temperature":
        dataset = DownscalingTemperatureDataset(args.dataset_path_hr, args.dataset_path_lr,
                                                max_len = args.dataset_size)
        dataloader = DataLoader(dataset, args.batch_size)
    elif args.dataset_type == "wind":
        dataset = DownscalingDataset(args.dataset_path_hr, args.dataset_path_lr,
                                     max_len = args.dataset_size)
        dataloader = DataLoader(dataset, args.batch_size)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    
# utils for parallelisation
def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def parallel_dataloader(rank, world_size, args, pin_memory=False, num_workers=0):
    batch_size = args.batch_size
    if args.dataset_type == "temperature":
        dataset = DownscalingTemperatureDataset(args.dataset_path_hr, args.dataset_path_lr,
                                                max_len = args.dataset_size)
    elif args.dataset_type == "wind":
        dataset = DownscalingDataset(args.dataset_path_hr, args.dataset_path_lr,
                                     max_len = args.dataset_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, 
                                 shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            pin_memory=pin_memory, num_workers=num_workers, 
                            drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader
