import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # si pas d'interface graphique
import matplotlib.pyplot as plt

from ddim_v2 import UNet
from ddim import q_sample, p_sample


def load_checkpoint(model, ckpt_path, device="cuda"):
    """
    Charge les poids stockés dans un .pth ou .pt.
    """
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

# ================================================
# 2) La fonction principale de génération
# ================================================
def generate_images(
    ckpt_path="model_ckpt_100.pth",
    T=1000,
    alphas=None,
    sigmas=None,
    num_images=5,
    img_size=28,
    device="cuda"
):
    """
    Charge un modèle, génère num_images échantillons,
    et les sauvegarde en png.
    """
    # 1) Créer le modèle
    model = UNet().to(device)
    model.eval()

    # 2) Charger le checkpoint
    model = load_checkpoint(model, ckpt_path, device)

    # 3) Générer des images
    for idx in range(num_images):
        # On part d'un bruit gaussien x_T
        x_t = torch.randn((1, 1, img_size, img_size), device=device)
        # Descente de T-1 à 0
        for i in reversed(range(T)):
            eps_theta = model(x_t, torch.tensor([i], device=device).long())
            x_t = p_sample(x_t, i, alphas, eps_theta[0], sigmas)

        # x_t est maintenant x_0 (théoriquement)
        final_img = x_t[0][0].detach().cpu().numpy()

        # 4) Sauvegarder l'image
        plt.figure()
        plt.imshow(final_img, cmap='gray')
        plt.axis('off')
        plt.savefig(f"generated_{idx}.png")
        plt.close()

# ================================================
# 3) Point d'entrée pour lancer dans le terminal
# ================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Suppose qu'on recrée le même schedule
    # qu'à l'entraînement :
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1. - betas
    sigmas = 0.0 * torch.ones_like(alphas)  # ex. DDIM déterministe

    generate_images(
        ckpt_path="model_ckpt.pth",
        T=T,
        alphas=alphas.to(device),
        sigmas=sigmas.to(device),
        num_images=5,
        img_size=28,
        device=device
    )
