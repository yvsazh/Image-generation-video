import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.utils as utils

# Параметри
BATCH_SIZE = 128
IMG_SIZE = 64
LATENT_DIM = 100
LEARNING_RATE = 0.0002
NUM_EPOCHS = 200
NUM_WORKERS = os.cpu_count() // 2

# Налаштування пристрою
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Трансформації для зображень
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Завантаження даних
data_path = "../GAN/datasets/faces/img_align_celeba"
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Покращений дискримінатор із spectral normalization та dropout
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # in: 3 x 64 x 64
            utils.spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32

            utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # out: 128 x 16 x 16

            utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # out: 256 x 8 x 8

            utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            # out: 512 x 4 x 4

            utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)),
            # out: 1 x 1 x 1
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)

# Покращений генератор з лінійним шаром для розгортання латентного вектору
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.init_size = IMG_SIZE // 16  # Наприклад, 4 якщо IMG_SIZE=64
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 512 * self.init_size * self.init_size))
        self.model = nn.Sequential(
            nn.BatchNorm2d(512),
            # in: 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # out: 256 x 8 x 8

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Вихідні значення в діапазоні від -1 до 1
            # out: 3 x 64 x 64
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 512, self.init_size, self.init_size)
        return self.model(out)

# Ініціалізація моделей
generator = Generator(LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

# Визначення функції втрат
adversarial_loss = nn.BCELoss()

# Оптимізатори
optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Шляхи до файлів з вагами
save_dir = "saved_models"
gen_path = os.path.join(save_dir, "generator.pth")
disc_path = os.path.join(save_dir, "discriminator.pth")

# Якщо збережені ваги існують, завантажуємо їх
if os.path.exists(gen_path) and os.path.exists(disc_path):
    generator.load_state_dict(torch.load(gen_path, map_location=device))
    discriminator.load_state_dict(torch.load(disc_path, map_location=device))
    print("Завантажено збережені ваги, продовжуємо навчання.")

# Папка для збереження згенерованих зображень
output_folder = "generated_images"
os.makedirs(output_folder, exist_ok=True)

# Тренувальний цикл з tqdm
for epoch in range(NUM_EPOCHS):
    epoch_loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
    for batch_idx, (real_imgs, _) in enumerate(epoch_loop):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Генерація випадкового шуму
        z = torch.randn(batch_size, LATENT_DIM, device=device)

        # ---------------------
        #  Тренування генератора
        # ---------------------
        optimizer_g.zero_grad()
        fake_imgs = generator(z)
        # Мітки "1" для обману дискримінатора
        valid = torch.ones(batch_size, 1, device=device)
        g_loss = adversarial_loss(discriminator(fake_imgs), valid)
        g_loss.backward()
        optimizer_g.step()

        # --------------------------
        #  Тренування дискримінатора
        # --------------------------
        optimizer_d.zero_grad()
        # Втрата для реальних зображень
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        # Втрата для фейкових зображень
        fake = torch.zeros(batch_size, 1, device=device)
        fake_loss = adversarial_loss(discriminator(generator(z).detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_d.step()

        # Оновлення tqdm опису батча
        epoch_loop.set_postfix({
            "Loss D": f"{d_loss.item():.4f}",
            "Loss G": f"{g_loss.item():.4f}"
        })

    # Збереження згенерованих зображень після кожної епохи
    with torch.no_grad():
        sample_z = torch.randn(16, LATENT_DIM, device=device)
        generated = generator(sample_z).detach().cpu()
    grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
    save_path = os.path.join(output_folder, f"img_epoch_{epoch+1}.png")
    torchvision.utils.save_image(grid, save_path)
    print(f"Збережено зображення: {save_path}")

# Після завершення тренування зберігаємо ваги моделей
os.makedirs(save_dir, exist_ok=True)
torch.save(generator.state_dict(), gen_path)
torch.save(discriminator.state_dict(), disc_path)
print("Ваги моделей успішно збережено!")
