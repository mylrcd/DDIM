import torch
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm

from ddim import q_sample, loss_function, sample_image


class DiffusionTrainer:
    def __init__(self, model, optimizer, train_loader, T, alphas, sigmas, device="cuda", img_size=28, num_channels = 1, dataset_name = "MNIST", display_color = 'gray', sample_interval=2):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.T = T
        self.alphas = alphas.to(device)
        self.sigmas = sigmas.to(device)
        self.img_size = img_size
        self.num_channels = num_channels
        self.dataset_name = dataset_name
        self.display_color = display_color
        self.sample_interval = sample_interval


    def train(self, epochs):
        dataset_size = len(self.train_loader.dataset)
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for images, _ in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images = images.to(self.device)
                batch_size = images.size(0)

                t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
                x_t, noise = q_sample(images, t, self.alphas, self.num_channels)
                noise_pred = self.model(x_t, t)
                loss = loss_function(noise_pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_size

            avg_train_loss = epoch_loss / dataset_size

            print(f"[Epoch {epoch+1}/{epochs}] Average Training Loss: {avg_train_loss:.4f}")

            if (epoch + 1) % self.sample_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    sample_image(self.T, self.img_size, self.alphas, self.sigmas, self.num_channels, self.dataset_name, self.display_color, self.model, epoch+1, self.device)
                self.model.train()  
            if (epoch + 1) % (self.sample_interval*10) == 0:
                self.model.eval()
                with torch.no_grad():
                    torch.save(self.model.state_dict(), f"models/model_{self.dataset_name}_ckpt_{epoch + 1}.pth")
                self.model.train() 