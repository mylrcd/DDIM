
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    
class Block(nn.Module):
    def __init__(self, input_channel, output_channel, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, output_channel)
        if up:
            self.conv1 = nn.Conv2d(2 * input_channel, output_channel, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(output_channel, output_channel, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
            self.transform = nn.Conv2d(output_channel, output_channel, 4, 2, 1)

        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(output_channel)
        self.bnorm2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, image, time):
        h = self.bnorm1(self.relu(self.conv1(image)))
        time_emb = self.relu(self.time_mlp(time))
        time_emb = time_emb[(...,) + (None,) * 2]  # Extend dimensions for addition
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)
    
    
    
class UNet(nn.Module):
    def __init__(self, im_channel):
        super().__init__()
        image_channel = im_channel  # Black-and-white images or color images
        down_channel = (64, 128, 256, 512, 1024)
        up_channel = (1024, 512, 256, 128, 64)
        out_dim = im_channel
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channel, down_channel[0], kernel_size=3, padding=1)

        # Downsample
        self.down = nn.ModuleList([
            Block(down_channel[i], down_channel[i + 1], time_emb_dim)
            for i in range(len(down_channel) - 1)
        ])

        # Upsample
        self.up = nn.ModuleList([
            Block(up_channel[i], up_channel[i + 1], time_emb_dim, up=True)
            for i in range(len(up_channel) - 1)
        ])

        # Final output layer
        self.output = nn.Conv2d(up_channel[-1], out_dim, 1)

    def forward(self, images, timestep):
        
        t = self.time_mlp(timestep)

        images = self.conv0(images)

        residual_inputs = []

        for down in self.down:
            images = down(images, t)
            residual_inputs.append(images)

        for up in self.up:
            residual_images = residual_inputs.pop()
            diff_h = residual_images.size(2) - images.size(2)
            diff_w = residual_images.size(3) - images.size(3)

            if diff_h > 0 or diff_w > 0:
                images = F.pad(images, (0, diff_w, 0, diff_h))
            elif diff_h < 0 or diff_w < 0:
                residual_images = residual_images[:, :, :images.size(2), :images.size(3)]
            images = torch.cat((images, residual_images), dim=1)
            images = up(images, t)
        output = self.output(images)
        return output