import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
import torchvision.transforms as T
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
from torchvision.io.image import ImageReadMode
from torchvision.io import read_image
import torch.nn.functional as F

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
            image_lr = self.transform_lr(image_lr)
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
            image_lr = self.transform_lr(image_lr)
        return image_hr, image_lr

class DownscalingMNIST(datasets.MNIST):
    def __init__(self, path, max_len = 1e6, **kwargs):
        super().__init__(path, transform = T.Compose([T.ToTensor(),
                                                      torchvision.transforms.Normalize((0.5), (0.5))]),
                                                     **kwargs)
        self.max_len = max_len

    def __len__(self):
        return np.min([super().__len__(), self.max_len])

    def __getitem__(self, idx):
        y, _ = super().__getitem__(idx)
        y = T.Resize((32,32))(y)
        x = T.Resize((8, 8))(y)
        return y, x

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
        norm = 255 / 2.0
        if args.image_size > 64:
            transform_hr = T.Compose([
                T.Resize((args.image_size, args.image_size)),
                T.Normalize((norm, norm, norm), (norm, norm, norm))
                ])
            transform_lr = T.Compose([
                T.Resize((args.image_size // 4, args.image_size // 4)),
                T.Normalize((norm, norm, norm), (norm, norm, norm))
                ])
        else:
            transform_hr = T.Compose([
                T.CenterCrop((args.image_size, args.image_size)),
                T.Normalize((norm, norm, norm), (norm, norm, norm))
                ])
            transform_lr = T.Compose([
                T.CenterCrop((args.image_size // 4, args.image_size // 4)),
                T.Normalize((norm, norm, norm), (norm, norm, norm))
                ])
        dataset = DownscalingTemperatureDataset(args.dataset_path_hr, args.dataset_path_lr,
                                                max_len = args.dataset_size,
                                                transform_hr = transform_hr,
                                                transform_lr = transform_lr)
    elif args.dataset_type == "wind":
        norm = 255 / 2.0
        if args.image_size > 64:
            transform_hr = T.Compose([
                T.Resize((args.image_size, args.image_size)),
                T.Normalize((norm, norm, norm), (norm, norm, norm))
                ])
            transform_lr = T.Compose([
                T.Resize((args.image_size // 4, args.image_size // 4)),
                T.Normalize((norm, norm, norm), (norm, norm, norm))
                ])
        else:
            transform_hr = T.Compose([
                T.CenterCrop((args.image_size, args.image_size)),
                T.Normalize((norm, norm, norm), (norm, norm, norm))
                ])
            transform_lr = T.Compose([
                T.CenterCrop((args.image_size // 4, args.image_size // 4)),
                T.Normalize((norm, norm, norm), (norm, norm, norm))
                ])
        dataset = DownscalingDataset(args.dataset_path_hr, args.dataset_path_lr,
                                     max_len = args.dataset_size,
                                     transform_hr = transform_hr,
                                     transform_lr = transform_lr)
    elif args.dataset_type == "MNIST":
        # currently only implemented for sizes 32 and 8
        dataset = DownscalingMNIST(args.dataset_path, max_len = args.dataset_size)
    if args.repeat_observations > 1:
        dataset = Subset(dataset, np.tile(np.arange(len(dataset)), args.repeat_observations))

    n_train = int(np.floor(len(dataset) / args.proportion_train))
    dataset_train = Subset(dataset, args.perm[0:n_train])
    dataset_test = Subset(dataset, args.perm[n_train:])

    dataloader_train = DataLoader(dataset_train, args.batch_size, shuffle=args.shuffle)
    dataloader_test = DataLoader(dataset_test, args.batch_size, shuffle=args.shuffle)
    return dataloader_train, dataloader_test

def get_data_simple(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
