
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
    def lr_preprocess_1(self, y, in_dim, out_dim):
        y = torch.flatten(y, 1)
        linear = nn.Linear(3*in_dim**2, self.out_channels*out_dim**2)
        y = linear(y)
        y = y.reshape(y.shape[0], self.out_channels, out_dim, out_dim)
        
        return y

    def forward(self, x, t, y=None):
        #add before maxpool_conv
        '''
        img_size = x.shape[-1]
        y = self.lr_preprocess_1(y, out_dim=img_size)
        x=x+y
        '''
        x = self.maxpool_conv(x)
        #add after maxpool_conv
        '''
        img_size = x.shape[-1]
        y = self.lr_preprocess_1(y, out_dim=img_size)
        x=x+y
        '''
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
    def lr_preprocess_1(self, y, in_dim, out_dim):
        y = torch.flatten(y, 1)
        linear = nn.Linear(3*in_dim**2, self.out_channels*out_dim**2)
        y = linear(y)
        y = y.reshape(y.shape[0], self.out_channels, out_dim, out_dim)
        
        return y


    def forward(self, x, skip_x, t, y=None):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        '''
        img_size = x.shape[-1]
        y = self.lr_preprocess_1(y, out_dim=img_size)
        x=x+y
        '''
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


'''class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output'''


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        
        #self.l1 = nn.Linear(3*256,3*64**2)
        #self.flatten=nn.Flatten(start_dim=2, end_dim=3)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
            
    #From (batch_size, 3, in_dim, in_dim) flatten to (batch_size, 3*in_dim_in_dim)
    #linear layer (batch_size, 3*in_dim*in_dim) to (batch_size, 3*out_dim*out_dim)
    #Reshape to (batch_size, 3, out_dim, out_dim)
    #Can add to high_resolution image at the start.
    #Can add to HR at each down/up sampling process.
    def lr_preprocess_1(self, y, in_dim, out_dim, in_channels, out_channels):
        y = torch.flatten(y, 1)
        linear = nn.Linear(in_channels*in_dim**2, out_channels*out_dim**2)
        y = linear(y)
        y = y.reshape(y.shape[0], out_channels, out_dim, out_dim)
        
        return y
    
    #Similar to lr_preprocess_1, but process the 3 channels respectively. 
    #Can add to high_resolution image at the start.
    #Can flatten to (batch_size, 256) and add to t at the beginning of the Unet.
    def lr_preprocess_2(self, y, in_dim, out_dim, in_channels, out_channels):
        b = y[:,0,:,:]
        g = y[:,1,:,:]
        r = y[:,2,:,:]
        b = torch.flatten(b,1)
        g = torch.flatten(g,1)
        r = torch.flatten(r,1)
        linear = nn.Linear(in_dim**2,out_dim**2)
        b=linear(b)
        g=linear(g)
        r=linear(r)
        b=b.reshape((b.shape[0],1,out_dim,out_dim))
        g=g.reshape((g.shape[0],1,out_dim,out_dim))
        r=r.reshape((r.shape[0],1,out_dim,out_dim))
        result=torch.cat((b,g,r),1)
        
        return result
        
            
         
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        '''
        y1 = self.lr_preprocess_1(y, 16, 64, 3, 3)
        x += y1
        '''
        '''
        y2 = self.lr_preprocess_2(y, 16, 64, 3, 3)
        x += y2
        '''

        #if y is not None:
            #t += self.label_emb(y)
        
        y2 = self.lr_preprocess_2(y, 16, 16, 3, 3)
        y2 = y2.mean(dim=1)
        y2 = torch.flatten(y2, 1)
        t += y2
        

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output




if __name__ == '__main__':
    
    net = UNet(device="cuda")
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    print('output shape')
    print(net(x,t).shape)
    net = UNet(device="cuda")
    net = UNet_downscale(device="cuda")
    print('number of parameters')
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = torch.randn(3, 3, 16, 16)
    print('output shape')
    print(net(x, t, y).shape)


    
    from utils import *
    dataloader = get_data(args)
    
    it = iter(dataloader)
    for i in range(10):
        x, y = next(it)
        t = x.new_tensor([500] * x.shape[0]).long()
        print(net(x, t, y).shape)
