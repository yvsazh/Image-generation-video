import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# ==== Визначення пристрою ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== Завантаження CelebA ====
data_path = "../GAN/datasets/faces/img_align_celeba"
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

celeba_data = datasets.ImageFolder(root=data_path, transform=transform)
data_loader = DataLoader(dataset=celeba_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# ==== Модель ====
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(64*64*3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 64*64*3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==== Тренування ====
def train_autoencoder(model, epochs=10, save_path="autoencoder_celeba.pth"):
    print("TRAINING STARTED")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} started")
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        for (img, _) in progress_bar:
            img = img.view(-1, 64*64*3).to(device, non_blocking=True)
            predictions = model(img)
            loss = criterion(predictions, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    # Зберігаємо модель
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ==== Генерація зображень ====
def generate_images(model_path="autoencoder_celeba.pth", num_images=8):
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_images, 10).to(device)  # Випадкові точки в латентному просторі
        generated = model.decoder(z).cpu()

    # Виводимо зображення та латентні вектори
    fig, axes = plt.subplots(2, num_images, figsize=(12, 6))
    for i in range(num_images):
        img = generated[i].view(3, 64, 64).numpy().transpose(1, 2, 0)  # Фікс осей
        
        # Візуалізація латентного вектора
        axes[0, i].bar(range(10), z[i].cpu().numpy())
        axes[0, i].set_ylim(-3, 3)
        
        # Виведення зображення
        axes[1, i].imshow(img)
        axes[1, i].axis("off")
    
    plt.show()

# ==== Запуск ====
if not os.path.exists("autoencoder_celeba.pth"):
    model = Autoencoder()
    train_autoencoder(model, epochs=10, save_path="autoencoder_celeba.pth")

generate_images(num_images=2)