import torch

import math
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

#visualisation dataset
def visualize_dataset(dataset, num_images=16):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if i >= num_images:
            break
        image, label = dataset[i]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
# forward diffusion
def q_sample(batch, t, alphas, batch_size):
    noise = torch.randn_like(batch)
    sqrt_alpha_t = torch.sqrt(alphas[t]).view(batch_size, 1, 1, 1)
    sqrt_one_minus_alpha_t = torch.sqrt((1 - alphas)[t]).view(batch_size, 1, 1, 1)
    x_t = sqrt_alpha_t * batch + sqrt_one_minus_alpha_t * noise
    return x_t, noise

# visualisation diffusion
def visualize_diffusion_step(noise_batch, alphas, num_steps, batch_size):
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 15))
    for step in range(num_steps):
        t = int(len(alphas) * step / 10)
        t = torch.randint(t, t+1, (batch_size,), device=device).long()
        noise_batch= q_sample(noise_batch, t, alphas, batch_size)
        x_t = noise_batch[0][0]
        axes[step].imshow(x_t, cmap="gray")
        axes[step].axis('off')

    plt.show()
    
    
    
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
    def __init__(self):
        super().__init__()
        image_channel = 1  # Black-and-white images
        down_channel = (64, 128, 256, 512)
        up_channel = (512, 256, 128, 64)
        out_dim = 1
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
        # Time embedding
        t = self.time_mlp(timestep)

        # Initial convolution
        images = self.conv0(images)

        residual_inputs = []

        # Downsampling
        for down in self.down:
            images = down(images, t)
            residual_inputs.append(images)

        # Upsampling
        for up in self.up:
            residual_images = residual_inputs.pop()

            # Adjust dimensions to match
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
    
    
def loss_function(predicted_noise, real_noise):
    return F.mse_loss(predicted_noise, real_noise)


@torch.no_grad()
def f_teta(x_t, alpha_t, eps_theta) :
    return (x_t - torch.sqrt(1.-alpha_t) * eps_theta) / torch.sqrt(alpha_t)


@torch.no_grad()
def p_sample(x_t, t, alphas, eps_theta, sigmas) :
    alpha_t = alphas[t]
    alpha_t_1 = alphas[t-1]
    sigma_t = sigmas[t]
    x0_pred = f_teta(x_t, alpha_t, eps_theta)
    if t > 1 :
        mean = ((torch.sqrt(alpha_t_1)) * x0_pred +
            torch.sqrt(1 - alpha_t_1 - sigma_t**2 ) *
            (x_t - torch.sqrt(alpha_t) * x0_pred) /
            torch.sqrt(alpha_t)
            )
        noise = sigma_t * torch.randn_like(x_t)
        x_t_1 = mean + noise

    else :
        # si t == 1
        mean = x0_pred
        noise = sigma_t * torch.randn_like(x_t)
        x_t_1 = mean + noise

    return x_t_1

@torch.no_grad()
def ddim_sample(x_T, t, alphas, eps_theta, sigmas, num_step) :

    x_t= x_T
    for t in range(num_step) :
        x_t = p_sample(x_t, t, alphas, eps_theta, sigmas)
    return x_t


@torch.no_grad()
def sample_plot_image(IMG_SIZE, T, alphas, sigmas, model, epoch):
    img = torch.randn((1, 1, IMG_SIZE, IMG_SIZE), device=device) 
    num_images = 11
    stepsize = T // (num_images-1)
    
    plt.figure(figsize=(15, 15))
    plt.subplot(1, num_images, 1)
    plt.imshow(img[0][0].detach().cpu().numpy(), cmap='gray')
    plt.title(f"step {0}", fontsize=8)
    plt.axis('off')
    #for i in range(T - 1, -1, -1):
    for i in reversed(range(T)):
        t = torch.randint(i, i+1, (1,), device=device).long()
        eps_theta = model(img, t)
        x_t = img[0][0]
        x_t_1 = p_sample(x_t, i, alphas, eps_theta[0][0], sigmas)
        x_t_1 = torch.clamp(x_t_1, -1.0, 1.0)
        img[0][0] = x_t_1
        if i % stepsize == 0:
            index = (T - 1 - i) // stepsize + 1
            plt.subplot(1, num_images, index+1)
            #plt.subplot(1, num_images, T // stepsize - i // stepsize)
            plt.imshow(x_t_1.detach().cpu().numpy(), cmap='gray')
            plt.title(f"step {T-i}", fontsize=8)
            plt.axis('off')
    #plt.show()
    plt.savefig(f"sample_epoch_{epoch}.png")
    plt.close()