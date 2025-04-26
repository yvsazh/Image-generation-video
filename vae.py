import torch
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Визначення пристрою
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Параметри
INPUT_DIM = 64 * 64 * 3  # CelebA розмір зображень 64x64x3
LATENT_DIM = 100
BATCH_SIZE = 128
LR_RATE = 1e-4
NUM_EPOCHS = 120

# Завантаження CelebA
data_path = "../GAN/datasets/faces/img_align_celeba"
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
celeba_data = datasets.ImageFolder(root=data_path, transform=transform)
train_loader = DataLoader(dataset=celeba_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Визначення VAE
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        reconstructed = self.decode(z)
        return reconstructed, mu, sigma

# Ініціалізація моделі
model = VariationalAutoEncoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.MSELoss(reduction="sum")

# Функція візуалізації
def plot_reconstructions(original, reconstructed, epoch, num_images=8):
    original = original[:num_images]
    reconstructed = reconstructed[:num_images]
    fig, axs = plt.subplots(2, num_images, figsize=(num_images*2, 4))
    for i in range(num_images):
        axs[0, i].imshow(original[i].cpu().permute(1, 2, 0))
        axs[0, i].axis('off')
        axs[1, i].imshow(reconstructed[i].cpu().permute(1, 2, 0))
        axs[1, i].axis('off')
    plt.suptitle(f'Epoch {epoch+1}: Top - original, Bottom - reconstructed')
    plt.show()

# Навчання
model.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = images.to(DEVICE).view(images.shape[0], -1)
        x_reconstructed, mu, sigma = model(images)
        reconstruction_loss = loss_fn(x_reconstructed, images)
        kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        loss = reconstruction_loss + kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {epoch_loss:.2f}")
# Збереження моделі
torch.save(model.state_dict(), "vae_celeba.pth")

# Генерація нових зображень
model.eval()
with torch.no_grad():
    random_latents = torch.randn(2, LATENT_DIM).to(DEVICE)
    generated_images = model.decode(random_latents).view(-1, 3, 64, 64).cpu()
    save_image(generated_images, "generated_celeba.png")
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for i in range(2):
        axs[i].imshow(generated_images[i].permute(1, 2, 0))
        axs[i].axis("off")
    plt.suptitle("Generated Images")
    plt.show()
