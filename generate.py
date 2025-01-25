import torch
import matplotlib
matplotlib.use('Agg')  # si pas d'interface graphique
import matplotlib.pyplot as plt
import argparse

from unet import UNet
from ddim import sample_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
T = 1000
N = 50
LEARNING_RATE = 1e-3
BETAS_START = 1e-4
BETAS_END = 0.02
ETA = 0
num_channels = 1

def setup_diffusion_params(T, device, eta):
    betas = torch.linspace(BETAS_START, BETAS_END, T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sigmas = eta * torch.sqrt(1. - alphas_cumprod).to(device)
    return alphas_cumprod, sigmas

def generate_images(
    T=1000,
    alphas=None,
    sigmas=None,
    N = 50,
    img_size=28,
    dataset_name="MNIST",
    device="cuda"
):
    ckpt_path="models/model_MNIST_ckpt_50.pth" if dataset_name == "MNIST" else "models/model_CIFAR_ckpt_200.pth"
    num_channels = 1 if dataset_name == "MNIST" else 3
    img_size = 28 if dataset_name == "MNIST" else 32
    display_color = 'gray' if dataset_name == "MNIST" else None
    model = UNet(num_channels).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)) #map_location=torch.device('cpu')
    model.eval()

    sample_image(T, img_size, alphas, sigmas, N, num_channels, f"TEST_{dataset_name}", display_color, model, 1, device)
    
        
if __name__ == "__main__":
    #python script.py --dataset MNIST
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé : {device}")

    alphas, sigmas = setup_diffusion_params(T, DEVICE, ETA)
    
    parser = argparse.ArgumentParser(description="Entraînez un modèle de diffusion sur MNIST ou CIFAR-10.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["MNIST", "CIFAR"], 
        help="Nom du dataset à entraîner (MNIST ou CIFAR)."
    )

    args = parser.parse_args()
    
    try:
        generate_images(
            T=T,
            alphas=alphas.to(device),
            sigmas=sigmas.to(device),
            N = N,
            img_size=28,
            dataset_name=args.dataset,
            device=device
        )
        print("Images générées avec succès.")
    except Exception as e:
        print(f"Erreur lors de la génération des images : {e}")