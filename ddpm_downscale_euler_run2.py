import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules_run2 import *
import logging
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import random

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672;
        compare also https://huggingface.co/blog/annotated-diffusion for other schedules
        """
        t = torch.linspace(0, self.noise_steps, self.noise_steps + 1)
        ft = torch.cos((t / self.noise_steps + 0.008) / 1.008 * np.pi / 2)**2
        alphat = ft / ft[0]
        betat = 1 - alphat[1:] / alphat[:-1]
        return torch.clip(betat, 0.0001, 0.9999)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, images_lr, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            images_lr = images_lr.to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, images_lr)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_downscale(interp_mode=args.interp_mode, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device, noise_steps=args.noise_steps)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images_hr, images_lr) in enumerate(pbar):
            images_hr = images_hr.to(device)
            images_lr = images_lr.to(device)
            t = diffusion.sample_timesteps(images_hr.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images_hr, t)
            predicted_noise = model(x_t, t, images_lr)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 50 == 0:
            # labels = torch.arange(10).long().to(device)
            # generate some random low-res images
            random_file=random.choice(os.listdir(args.dataset_path_lr))
            path =  os.path.join(args.dataset_path_lr, random_file)
            images_lr = read_image(path, mode = ImageReadMode(3)).unsqueeze(0)
            for i in range(args.n_example_imgs):
                random_file=random.choice(os.listdir(args.dataset_path_lr))
                path =  os.path.join(args.dataset_path_lr, random_file)
                random_img = read_image(path, mode = ImageReadMode(3)).unsqueeze(0)
                images_lr = torch.cat([images_lr, random_img], dim=0)

            sampled_images = diffusion.sample(model, n=len(images_lr), images_lr = images_lr)
            ema_sampled_images = diffusion.sample(ema_model, n=len(images_lr), images_lr=images_lr)
            plot_images(sampled_images)
            save_images(images_lr, os.path.join("results", args.run_name, f"{epoch}_lowres.jpg"))
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_downscale_1000datapoints"
    args.epochs = 300 #todo
    args.batch_size = 5 #todo
    args.image_size = 64
    args.interp_mode = 'bicubic'
    args.noise_steps = 750
    args.dataset_type = "wind"
    args.dataset_path_hr = "/cluster/work/math/climate-downscaling/WiSoSuper_data/train/wind/middle_patch/HR"
    args.dataset_path_lr = "/cluster/work/math/climate-downscaling/WiSoSuper_data/train/wind/middle_patch/LR"
    #args.dataset_path_hr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch_subset/HR"
    #args.dataset_path_lr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch_subset/LR"
    args.device = "cuda" #todo
    args.lr = 9e-4
    args.n_example_imgs = 4 #todo
    args.dataset_size = 1000
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1000"
    train(args)


if __name__ == '__main__':
    # pass
    launch()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)
