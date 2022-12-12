#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:34:13 2022

@author: mschillinger
"""

from ddpm import *
from modules import * 
from utils import *
import os

if __name__ == '__main__': 
    device = "cpu"
    model = UNet(device=device).to(device)
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirname, "checkpoints/unconditional_ckpt.pt")
    ckpt = torch.load(filename, map_location=torch.device('cpu') )
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, n=1)
    plot_images(x)

