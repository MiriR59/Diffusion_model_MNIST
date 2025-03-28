import torch
import torch.nn as nn # Building block for NN
import torch.optim as optim # Optimization algorithms
import torchvision # Image-related utility
import torchvision.transforms as transforms # Image to tensor transfer utility
import torch.nn.functional as F
import matplotlib.pyplot as plt # Plotting
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- GPU check ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 

# --- Hyperparameters ---
T = 100                # Diffusion steps
beta_start = 0.001    # Noise variance start
beta_end = 0.02         # Final noise variance
image_size = 28
batch_size = 64
learning_rate = 0.0001 # LEARNING RATE !!!!!!!!!!!
epochs = 350
num_samples = 5
losses = []

# --- Define noise-adding ---
beta = torch.linspace(beta_start, beta_end, T).to(device)
alpha = 1 - beta
alpha_c = torch.cumprod(alpha, dim=0)

# --- Forward Diffusion Process ---
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0).to(device)

    s_alpha_c = alpha_c[t].view(-1, 1, 1, 1).sqrt()
    s_1_alpha_c = (1 - alpha_c[t]).view(-1, 1, 1, 1).sqrt()
    
    xt = s_alpha_c * x0 + s_1_alpha_c * noise
    
    return xt, noise

# --- Architecture ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder - spatial reduction
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU())
        
        self.bottleneck1 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU())
        self.bottleneck2 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU())
        
        # Decoder - spatial back to original
        self.dec4 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1))
        
        # Time embedding
        self.time_embed = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 128))
        
    def forward(self, x, t):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        embed = self.time_embed(t.float().unsqueeze(1)).view(-1, 128, 1, 1)
        embed = embed.expand(-1, -1, e4.shape[2], e4.shape[3])
        
        e4 = torch.cat([e4, embed], dim=1)
        e4 = nn.LayerNorm(e4.shape[1:], elementwise_affine=False)(e4)
        
        b1 = self.bottleneck1(e4)
        b2 = self.bottleneck2(b1)
        
        d4 = self.dec4(b2)
        d3 = self.dec3(d4) + e2     # Skip con
        d2 = self.dec2(d3) + e1     # Skip con
        d1 = self.dec1(d2)
        
        return d1
    
# Sampling new images (reverse diffusion process)
def new_images(model, num_samples=num_samples, image_size=image_size, T=T):
    x = torch.randn((num_samples, 1, image_size, image_size), device=device)
    
    for t in reversed(range(T)):
        with torch.no_grad():
            t_tensor = torch.full((num_samples,), t, device=device)
            noise_pred = model(x, t_tensor)
            
            beta_t = beta[t]
            alpha_t = alpha[t]
            alpha_cum = alpha_c[t]
            
            if t > 0:
                noise = torch.randn_like(x).to(device)
            else:
                noise = torch.zeros_like(x).to(device)

            # Reverse diffusion step
            x = (1 / alpha_t.sqrt()) * (x - beta_t * noise_pred / (1 - alpha_cum).sqrt()) + noise * beta_t.sqrt()

    return x.cpu()

# --- Dataset creation ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)

# --- Filter only images of the digit '1' ---
filtered_data = [(img, label) for img, label in dataset if label == 1]
filtered_data = random.sample(filtered_data, 200)
dataset_one = torch.utils.data.TensorDataset(torch.stack([img for img, _ in filtered_data]), torch.tensor([label for _, label in filtered_data]))
dataloader_one = torch.utils.data.DataLoader(dataset_one, batch_size=batch_size, shuffle=True)

# --- Filter only images of the digit 5' ---
# filtered_data = [(img, label) for img, label in dataset if label == 5]
# dataset_five = torch.utils.data.TensorDataset(torch.stack([img for img, _ in filtered_data]), torch.tensor([label for _, label in filtered_data]))
# dataloader_five = torch.utils.data.DataLoader(dataset_five, batch_size=batch_size, shuffle=True)

# --- Small dataset ---
# random_samples = random.sample(filtered_data, 10)
# small_dataset = torch.utils.data.TensorDataset(torch.stack([img for img, _ in random_samples]), torch.tensor([label for _, label in random_samples]))
# small_dataloader = torch.utils.data.DataLoader(small_dataset, batch_size=batch_size, shuffle=True)

print(f"Number of images of '1': {len(dataset_one)}")

# --- Learning prep ---
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.9)
loss_fn = nn.MSELoss()

# --- Training Loop ---
for epoch in range(epochs):
    loss_c = 0
    num_batches = 0
    
    for images, _ in dataset_one:
        images = images.to(device)
        
        batch_size_actual = images.shape[0]
        t = torch.randint(0, T, (batch_size_actual,), device=device)
        xt, noise = forward_diffusion(images, t)
        noise = noise.unsqueeze(1)
        predicted_noise = model(xt, t)
        loss = loss_fn(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_c += loss
        num_batches += 1
        
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    loss_c /= num_batches
    losses.append(loss_c.item())
    print(f"Epoch {epoch+1}: Loss = {loss_c.item():.6f}, Learning Rate = {current_lr:.6e}")
    
    if (epoch + 1) % 10 == 0:

        # Generate and display new images
        samples = new_images(model)
        fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
        for i, ax in enumerate(axes):
            ax.imshow(samples[i].squeeze(), cmap="gray")
            ax.axis("off")
        plt.show()

    
torch.save(model.state_dict(), "diffusion_model.pth")
print("Model saved successfully.")

# --- Plot sampled images ---
samples = new_images(model)
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(samples[i].squeeze(), cmap="gray")
    ax.axis("off")
plt.show()

# --- Plot loss curve ---
plt.plot(losses, label="Total loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss curve during training")
plt.legend()
plt.show()