import torch
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Constants (same as training) ---
T = 2000
beta_start = 1e-5
beta_end = 7.5e-3
image_size = 28
num_samples = 1

# --- Noise schedule ---
beta = torch.linspace(beta_start, beta_end, T).to(device)
alpha = 1 - beta
alpha_cumulative = torch.cumprod(alpha, dim=0)

# --- Load the trained model ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU())
        
        self.bottleneck1 = nn.Sequential(nn.Conv2d(320, 320, kernel_size=3, padding=1), nn.ReLU())
        self.bottleneck2 = nn.Sequential(nn.Conv2d(320, 256, kernel_size=3, padding=1), nn.ReLU())
        
        self.dec4 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1))
        
        self.time_embed = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))
        
    def forward(self, x, t):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        embed = self.time_embed(t.float().unsqueeze(1)).view(-1, 64, 1, 1)
        embed = embed.expand(-1, -1, e4.shape[2], e4.shape[3])
        e4 = torch.cat([e4, embed], dim=1)

        b1 = self.bottleneck1(e4)
        b2 = self.bottleneck2(b1)
        d4 = self.dec4(b2)
        d3 = self.dec3(d4) + e2
        d2 = self.dec2(d3) + e1
        d1 = self.dec1(d2)
        
        return d1

model = Net().to(device)
model.load_state_dict(torch.load("diffusion_model.pth"))
model.eval()
print("Model loaded.")

# --- Load a clean image ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
img = Image.open('dataset_ones/1.png').convert('L')
img = transform(img).unsqueeze(0).to(device)  # Shape: [1, 1, 28, 28]

# --- Forward diffusion helper ---
def forward_diffusion(image, t):
    noise = torch.randn_like(image).to(device)
    root_image = alpha_cumulative[t].view(-1, 1, 1, 1).sqrt()
    root_noise = (1 - alpha_cumulative[t]).view(-1, 1, 1, 1).sqrt()
    xt = root_image * image + root_noise * noise
    return xt, noise

# --- Side test at multiple noise levels ---
steps = [int(p * T) for p in [0.1, 0.25, 0.5, 0.75, 0.95]]
fig, axs = plt.subplots(len(steps), 3, figsize=(10, 12))

# --- Reversing in one jump ---
for i, t in enumerate(steps):
    t_tensor = torch.tensor([t], device=device)
    xt, true_noise = forward_diffusion(img, t_tensor)
    
    with torch.no_grad():
        predicted_noise = model(xt, t_tensor)

    denoised = (xt - predicted_noise * (1 - alpha_cumulative[t]).sqrt().view(-1, 1, 1, 1)) / alpha_cumulative[t].sqrt().view(-1, 1, 1, 1)

    axs[i, 0].imshow(img.squeeze().cpu(), cmap='gray')
    axs[i, 0].set_title(f"Original")

    axs[i, 1].imshow(xt.squeeze().cpu(), cmap='gray')
    axs[i, 1].set_title(f"Noisy t={t}")

    axs[i, 2].imshow(denoised.squeeze().clamp(-1, 1).cpu(), cmap='gray')
    axs[i, 2].set_title(f"Denoised")

    for j in range(3):
        axs[i, j].axis('off')

plt.tight_layout()
plt.show()

# --- Reversing step-by-step ---
for i, t in enumerate(steps):
    t_tensor = torch.tensor([t], device=device)
    xt, true_noise = forward_diffusion(img, t_tensor)

    # Reverse denoising from xt to x0
    x = xt.clone()
    for rev_t in reversed(range(t + 1)):  # Go from t back to 0
        with torch.no_grad():
            t_rev_tensor = torch.full((num_samples,), rev_t, device=device)
            noise_pred = model(x, t_rev_tensor)

            beta_t = beta[rev_t]
            alpha_t = alpha[rev_t]
            alpha_c = alpha_cumulative[rev_t]

            noise = torch.randn_like(x) if rev_t > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t * noise_pred / torch.sqrt(1 - alpha_c))) + noise * torch.sqrt(beta_t)

    denoised = x

    axs[i, 0].imshow(img.squeeze().cpu(), cmap='gray')
    axs[i, 0].set_title(f"Original")

    axs[i, 1].imshow(xt.squeeze().cpu(), cmap='gray')
    axs[i, 1].set_title(f"Noisy t={t}")

    axs[i, 2].imshow(denoised.squeeze().clamp(-1, 1).cpu(), cmap='gray')
    axs[i, 2].set_title(f"Step-by-step Denoised")

    for j in range(3):
        axs[i, j].axis('off')

plt.tight_layout()
plt.show()
