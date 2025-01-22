import torch

import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"


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


class DiffusionTrainer:
    def __init__(self, model, optimizer, train_loader,
                 T, alphas, sigmas,
                 device="cuda", img_size=28, sample_interval=2):
        """
        Args:
            model: Le UNet (ou tout autre réseau) à entraîner.
            optimizer: Optimiseur PyTorch (Adam, etc.).
            train_loader: DataLoader pour fournir les batches.
            T: nombre de steps de diffusion (par ex. 1000).
            alphas: tenseur des alphas (size=T).
            sigmas: tenseur des sigmas (size=T).
            device: "cuda" ou "cpu".
            img_size: taille des images (H = W = 28 pour MNIST).
            sample_interval: tous les combien d'epochs on sauvegarde un échantillon.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.T = T
        self.alphas = alphas.to(device)
        self.sigmas = sigmas.to(device)
        self.img_size = img_size
        self.sample_interval = sample_interval

    def loss_function(self, predicted_noise, real_noise):
        """Simple MSE loss."""
        return F.mse_loss(predicted_noise, real_noise)

    def q_sample(self, x0, t):
        """Sample x_t given x_0 and noise, suivant q(x_t|x_0)."""
        noise = torch.randn_like(x0)
        sqrt_alpha_t = torch.sqrt(self.alphas[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1. - self.alphas[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise

    @torch.no_grad()
    def p_sample(self, x_t, t, eps_theta):
        """
        Etape de sampling x_{t-1} depuis x_t.
        x_t: shape (1, 1, H, W) si batch=1 (pour simplifier).
        eps_theta: bruit prédit par le modèle.
        """
        alpha_t = self.alphas[t]
        if t == 0:
            return (x_t - torch.sqrt(1. - alpha_t) * eps_theta) / torch.sqrt(alpha_t)

        alpha_t_1 = self.alphas[t-1]
        sigma_t = self.sigmas[t]

        # x0_pred = f_theta(x_t)
        x0_pred = (x_t - torch.sqrt(1. - alpha_t) * eps_theta) / torch.sqrt(alpha_t)

        mean = (
            torch.sqrt(alpha_t_1) * x0_pred
            + torch.sqrt(1. - alpha_t_1 - sigma_t**2)
              * (x_t - torch.sqrt(alpha_t) * x0_pred) / torch.sqrt(alpha_t)
        )
        noise = sigma_t * torch.randn_like(x_t)
        return mean + noise

    @torch.no_grad()
    def sample_image(self, epoch):
        T = self.T
        img = torch.randn((1, 1, self.img_size, self.img_size), device=self.device)
        num_images = 10
        stepsize = max(1, T // (num_images - 1))

        plt.figure(figsize=(15, 3))
        plt.subplot(1, num_images, 1)
        plt.imshow(img[0][0].detach().cpu().numpy(), cmap='gray')
        plt.title("x_T")
        plt.axis('off')

        for i in reversed(range(T)):
            eps_theta = self.model(img, torch.tensor([i], device=self.device).long())
            img = self.p_sample(img, i, eps_theta)

            if i % stepsize == 0:
                idx = (T - i) // stepsize + 1
                plt.subplot(1, num_images, idx)
                plt.imshow(img[0][0].detach().cpu().numpy(), cmap='gray')
                plt.title(f"t={i}")
                plt.axis('off')

        plt.tight_layout()
        plt.savefig(f"sample_epoch_{epoch}.png")
        plt.close()

    def train(self, epochs):
        dataset_size = len(self.train_loader.dataset)
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for images, _ in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                batch_size = images.size(0)

                t = torch.randint(0, self.T, (batch_size,), device=self.device).long()

                x_t = self.q_sample(images, t)
                noise = torch.randn_like(images)

                noise_pred = self.model(x_t, t)

                loss = self.loss_function(noise_pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_size

            avg_train_loss = epoch_loss / dataset_size

            print(f"[Epoch {epoch+1}/{epochs}] Average Training Loss: {avg_train_loss:.4f}")

            if (epoch + 1) % self.sample_interval == 0:
                self.sample_image(epoch + 1)
            if (epoch + 1) % (self.sample_interval*10) == 0:
                torch.save(self.model.state_dict(), f"model_ckpt_{epoch}.pth")


def make_beta_schedule(num_timesteps, start=1e-4, end=0.02):
    """Exemple de schedule linéaire pour betas."""
    return torch.linspace(start, end, num_timesteps)

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    batch_size = 64
    img_size = 28
    T = 1000 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="/data",
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    betas = make_beta_schedule(T, start=1e-4, end=0.02)
    alphas = 1. - betas
    alphas = torch.cumprod(alphas, dim=0)
    eta = 0.5
    sigmas = eta * torch.sqrt(1. - alphas).to(device)

    #sigmas = 0.0 * torch.ones_like(alphas) samplit deterministe

    model = UNet()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    trainer = DiffusionTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        T=T,
        alphas=alphas,
        sigmas=sigmas,
        device=device,
        img_size=img_size,
        sample_interval=1
    )
    epochs = 100
    trainer.train(epochs=epochs)

if __name__ == "__main__":
    main()