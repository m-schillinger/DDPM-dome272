import os
import copy
import numpy as np
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

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, noise_schedule = "linear", \
    beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule(type = noise_schedule).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self, type):
        if type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif type == "cosine":
            # cosine schedule as proposed in https://arxiv.org/abs/2102.09672;
            # compare also https://huggingface.co/blog/annotated-diffusion for other schedules
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

    def sample(self, model, n, images_lr, c_in = 3, cfg_scale=0):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, c_in, self.img_size, self.img_size)).to(self.device)
            images_lr = images_lr.to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, images_lr)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                # save intermediate results
                # To do
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader, dataloader_test = get_data(args)
    model = UNet_downscale(c_in = args.c_in, c_out = args.c_out,
                           img_size = args.image_size,
                           interp_mode=args.interp_mode, device=device).to(device)
    model.load_state_dict(torch.load('./models/NewTransform_NewSampling_fixed_s-10000_train-0.5/DDPM_downscale_wind_ns-linear__bs-15_e-801_lr-0.0001_cfg0.1_size64_shuffleTrue/ckpt.pt',
             map_location=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(torch.load('./models/NewTransform_NewSampling_fixed_s-10000_train-0.5/DDPM_downscale_wind_ns-linear__bs-15_e-801_lr-0.0001_cfg0.1_size64_shuffleTrue/optim.pt',
             map_location=device))
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device, \
        noise_steps=args.noise_steps, noise_schedule=args.noise_schedule)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(368, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images_hr, images_lr) in enumerate(pbar):
            images_hr = images_hr.to(device)
            images_lr = images_lr.to(device)
            if np.random.random() < args.cfg_proportion:
                images_lr = None
            t = diffusion.sample_timesteps(images_hr.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images_hr, t)
            predicted_noise = model(x_t, t, images_lr)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 50 == 0:
            # labels = torch.arange(10).long().to(device)
            # generate some random low-res images
            if args.dataset_type == "wind" or args.dataset_type == "temperature":
                # alternative version: sample only from training data
                it = iter(dataloader)
                images_hr, images_lr = next(it)
                images_lr = images_lr[0:args.n_example_imgs]
                images_hr = images_hr[0:args.n_example_imgs]
                images_hr_save =  (images_hr + 1) / 2.0 * 255
                images_hr_save = images_hr_save.type(torch.uint8)
                images_lr_save =  (images_lr + 1) / 2.0 * 255
                images_lr_save = images_lr_save.type(torch.uint8)

                sampled_images = diffusion.sample(model, n=len(images_lr), images_lr = images_lr, cfg_scale = 0)
                save_images(images_lr_save, os.path.join("results", args.run_name, f"{epoch}_lowres.jpg"))
                save_images(images_hr_save, os.path.join("results", args.run_name, f"{epoch}_truth.jpg"))
                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}_generated.jpg"))
                torch.save(sampled_images, os.path.join("results", args.run_name, f"{epoch}_tensor.pt"))

                # sampled_images_cfg1 = diffusion.sample(model, n=len(images_lr), images_lr = images_lr_transformed, cfg_scale = 0.1)
                # sampled_images_cfg2 = diffusion.sample(model, n=len(images_lr), images_lr = images_lr_transformed, cfg_scale = 3)
                # ema_sampled_images = diffusion.sample(ema_model, n=len(images_lr), images_lr=images_lr_transformed)
                # save_images(sampled_images_cfg1, os.path.join("results", args.run_name, f"{epoch}_cfg0-1.jpg"))
                # save_images(sampled_images_cfg2, os.path.join("results", args.run_name, f"{epoch}_cfg3.jpg"))
                # save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))

                it_test = iter(dataloader_test)
                images_hr, images_lr = next(it_test)
                images_lr = images_lr[0:args.n_example_imgs]
                images_hr = images_hr[0:args.n_example_imgs]
                images_hr_save =  (images_hr + 1) / 2.0 * 255
                images_hr_save = images_hr_save.type(torch.uint8)
                images_lr_save =  (images_lr + 1) / 2.0 * 255
                images_lr_save = images_lr_save.type(torch.uint8)

                sampled_images = diffusion.sample(model, n=len(images_lr), images_lr = images_lr, cfg_scale = 0)
                save_images(images_lr_save, os.path.join("results", args.run_name, f"{epoch}_test_lowres.jpg"))
                save_images(images_hr_save, os.path.join("results", args.run_name, f"{epoch}_test_truth.jpg"))
                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}_test_generated.jpg"))
                torch.save(sampled_images, os.path.join("results", args.run_name, f"{epoch}_test_tensor.pt"))


            elif args.dataset_type == "MNIST":
                it = iter(dataloader)
                images_hr, images_lr = next(it)
                for i in range(args.n_example_imgs):
                    images_hr, random_img = next(it)
                    images_lr = torch.cat([images_lr, random_img], dim=0)

                sampled_images = diffusion.sample(model, n=len(images_lr), images_lr = images_lr, c_in =1, cfg_scale = 0)
                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
                grid = torchvision.utils.make_grid(images_lr)
                ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
                ndarr = (ndarr + 1 ) / 2.0 # rescale to 0,1
                plt.imsave(os.path.join("results", args.run_name, f"{epoch}_lowres.jpg"), ndarr, cmap = "gray")
                torch.save(sampled_images, os.path.join("results", args.run_name, f"{epoch}_tensor.pt"))

            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            # torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default = 15)
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
    args.proportion_train = 2.0
    if args.lr == 0.0:
        args.lr = 3e-4 * 14 / args.batch_size
    if args.dataset_type == "wind":
        args.dataset_path_hr = "/cluster/work/math/climate-downscaling/WiSoSuper_data/train/wind/middle_patch/HR"
        args.dataset_path_lr = "/cluster/work/math/climate-downscaling/WiSoSuper_data/train/wind/middle_patch/LR"
        args.c_in = 6
        args.c_out = 3
        if args.image_size is None:
            args.image_size = 64
    elif args.dataset_type == "temperature":
        args.dataset_path_lr = "/cluster/work/math/climate-downscaling/kba/tas_lowres_colour_widerange"
        args.dataset_path_hr = "/cluster/work/math/climate-downscaling/kba/tas_highres_colour_widerange"
        args.c_in = 6
        args.c_out = 3
        if args.image_size is None:
            args.image_size = 64
    elif args.dataset_type == "MNIST":
        args.dataset_path = "/cluster/home/mschillinger/DL-project/MNIST"
        args.c_in = 2
        args.c_out = 1
        if args.image_size is None:
            args.image_size = 32
    args.run_name = f"NewTransform_NewSampling_fixed_s-{args.dataset_size}_train-0.5/DDPM_downscale_{args.dataset_type}_ns-{args.noise_schedule}__bs-{args.batch_size}_e-{args.epochs}_lr-{args.lr}_cfg{args.cfg_proportion}_size{args.image_size}_shuffle{args.shuffle}"
    args.interp_mode = 'bicubic'
    args.noise_steps = 750
    args.device = "cuda"
    args.n_example_imgs = 5
    if args.dataset_size == 10000:
        # fixed random permutation in this case
        with open('data_permutation', 'rb') as data_permutation_file:
            args.perm = pickle.load(data_permutation_file)
    else:
        args.perm = np.random.permutation(np.arange(0, args.dataset_size,1))

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
    # x = diffusion.sample(model, n, y, le=0)
    # plot_images(x)