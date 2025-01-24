import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
    
@torch.no_grad()
def p_sample(x_t, t, alphas, sigmas, eps_theta):
    alpha_t = alphas[t]
    if t == 0:
        return (x_t - torch.sqrt(1. - alpha_t) * eps_theta) / torch.sqrt(alpha_t)

    alpha_t_1 = alphas[t-1]
    sigma_t = sigmas[t]
    
    x0_pred = (x_t - torch.sqrt(1. - alpha_t) * eps_theta) / torch.sqrt(alpha_t)
    x0_pred = torch.clamp(x0_pred, -1., 1.) 
    mean = (
        torch.sqrt(alpha_t_1) * x0_pred
        + torch.sqrt(1. - alpha_t_1 - sigma_t**2) * eps_theta
    )
    noise = sigma_t * torch.randn_like(x_t)
    return mean + noise

@torch.no_grad()
def sample_image(T, img_size, alphas, sigmas, num_channels, dataset_name, display_color, model, epoch, device):
    img = torch.randn((1, num_channels, img_size, img_size), device=device)
    img = img.permute(0, 2, 3, 1)
    img_display = img[0] * 0.5 + 0.5
    img_display = torch.clamp(img_display, 0., 1.)
    img = torch.clamp(img, -1., 1.)
    num_images = 11
    stepsize = max(1, T // (num_images-1))
    plt.figure(figsize=(15, 3))
    plt.subplot(1, num_images, 1)
    plt.imshow(img_display.detach().cpu().numpy(), cmap=display_color if display_color == 'gray' else None)
    plt.title("x_T")
    plt.axis('off') 
    with torch.no_grad():
        q=0
        for i in reversed(range(T)):
            img = img.permute(0, 3, 1, 2)
            
            eps_theta = model(img, torch.tensor([i], device=device).long())
            
            q+=1
            img = p_sample(img, i, alphas, sigmas, eps_theta)
            img = torch.clamp(img, -1., 1.)
            img = img.permute(0, 2, 3, 1)
            img_display = img[0] * 0.5 + 0.5
            img_display = torch.clamp(img_display, 0., 1.)
            if i % stepsize == 0:
                idx = (T - i) // stepsize + 1
                plt.subplot(1, num_images, idx)
                plt.imshow(img_display.detach().cpu().numpy(), cmap=display_color if display_color == 'gray' else None)
                
                plt.title(f"t={i}")
                plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"results/sample_{dataset_name}_epoch_{epoch}.png")
        plt.close()

# forward diffusion
def loss_function(predicted_noise, real_noise):
    return F.mse_loss(predicted_noise, real_noise)

def q_sample(x0, t, alphas, num_channels):
    """Sample x_t given x_0 and noise, suivant q(x_t|x_0)."""
    noise = torch.randn_like(x0)
    sqrt_alpha_t = torch.sqrt(alphas[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_t = torch.sqrt(1. - alphas[t]).view(-1, 1, 1, 1)
    x_t = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise
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