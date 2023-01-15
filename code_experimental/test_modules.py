from modules import *
net = UNet(device="cpu")
x = torch.randn(3, 3, 64, 64)
t = x.new_tensor([500] * x.shape[0]).long()
print('output shape')
print(net(x,t).shape)
# net = UNet(device="cpu")
net = UNet_downscale(device="cpu")
print('number of parameters')
print(sum([p.numel() for p in net.parameters()]))
x = torch.randn(3, 3, 64, 64)
t = x.new_tensor([500] * x.shape[0]).long()
y = torch.randn(3, 3, 16, 16)
print('output shape')
print(net(x, t, y).shape)

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.run_name = "DDPM_downscale"
args.epochs = 300
args.batch_size = 5
args.image_size = 64
args.dataset_type = "wind"
args.dataset_size = 100
args.dataset_path_hr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch/HR"
args.dataset_path_lr = "/scratch/users/mschillinger/Documents/DL-project/WiSoSuper/train/wind/middle_patch/LR"
args.device = "cpu"
args.lr = 3e-4

from utils import *
dataloader = get_data(args)

it = iter(dataloader)
for i in range(10):
   x, y = next(it)
   t = x.new_tensor([500] * x.shape[0]).long()
   print(net(x, t, y).shape)
