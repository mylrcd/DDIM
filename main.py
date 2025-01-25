import torch
import matplotlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
import argparse

from unet import UNet
from trainer import DiffusionTrainer

matplotlib.use('Agg') 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
T = 1000
LEARNING_RATE = 1e-3
BETAS_START = 1e-4
BETAS_END = 0.02
ETA = 0
N = 50


def get_dataloader(dataset_name, batch_size):
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # Convertit [0, 1] -> [-1, 1]
        ])
        dataset = datasets.MNIST(
            root=".././data",
            train=True,
            download=True,
            transform=transform
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif dataset_name == "CIFAR":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(
            root=".././data/CIFAR/",
            train=True,
            download=True,
            transform=transform
        )
        
        plane_indices = [i for i, label in enumerate(dataset.targets) if label == 0]
        subset = Subset(dataset, plane_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)

    else:
        raise ValueError(f"Dataset {dataset_name} non pris en charge.")

def setup_diffusion_params(T, device, eta):
    betas = torch.linspace(BETAS_START, BETAS_END, T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    #sigmas = eta * torch.sqrt(1. - alphas_cumprod).to(device)
    tau = torch.arange(0, T, device=device)
    sigmas = torch.rand_like(alphas)
    sigmas[0] = 1.e-2
    sigmas[1:] = ETA * torch.sqrt((1 - alphas_cumprod[tau[:-1]]) / (1 - alphas_cumprod[tau[1:]])) \
            * torch.sqrt((1 - alphas_cumprod[tau[1:]]) / alphas_cumprod[tau[:-1]])
    sigmas = sigmas.to(device)
    return alphas_cumprod, sigmas

def train_diffusion_model(dataset_name, epochs=1):
    print(f"Entraînement sur {dataset_name} avec l'appareil : {DEVICE}")
    
    num_channels = 1 if dataset_name == "MNIST" else 3
    img_size = 28 if dataset_name == "MNIST" else 32
    batch_size = 64 if dataset_name == "MNIST" else 8
    display_color = 'gray' if dataset_name == "MNIST" else None
    train_loader = get_dataloader(dataset_name, batch_size)
    alphas, sigmas = setup_diffusion_params(T, DEVICE, ETA)

    model = UNet(num_channels).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = DiffusionTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        T=T,
        alphas=alphas,
        sigmas=sigmas,
        N = N,
        device=DEVICE,
        img_size=img_size,
        num_channels = num_channels,
        dataset_name = dataset_name,
        display_color = display_color,
        sample_interval=1
    )

    trainer.train(epochs=epochs)


if __name__ == "__main__":
    #python script.py --dataset MNIST --epochs 5
    parser = argparse.ArgumentParser(description="Entraînez un modèle de diffusion sur MNIST ou CIFAR-10.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["MNIST", "CIFAR"], 
        help="Nom du dataset à entraîner (MNIST ou CIFAR)."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1, 
        help="Nombre d'époques pour l'entraînement."
    )
    args = parser.parse_args()
    train_diffusion_model(dataset_name=args.dataset, epochs=args.epochs)