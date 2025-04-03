import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms 
import torch.nn.init as init
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- GPU check ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 

def images_to_list(image_folder):
    # Custom Dataset class to load images and convert them to arrays
    class CustomImageDataset(Dataset):
        def __init__(self, image_folder, transform=None):
            self.image_folder = image_folder
            self.transform = transform
            # List all PNG files in the image folder
            self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
            
        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            # Open the image as grayscale
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            
            if self.transform:
                img = self.transform(img)  # Apply transformations
            
            # Convert the image to a NumPy array and remove the channel dimension
            img_np = img.squeeze().cpu().numpy()  # Shape (28, 28)
            
            return img_np  # Return as NumPy array

    # Define transformation (ToTensor and Normalize)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the dataset (images from the 'image_folder')
    dataset = CustomImageDataset(image_folder=image_folder, transform=transform)

    # Get list of NumPy arrays representing the images
    image_list = [dataset[i] for i in range(len(dataset))]

    # Convert list of arrays to a 4D tensor
    image_tensor = torch.tensor(np.array(image_list)).unsqueeze(1).float()

    return image_tensor

    print('Dataset created')
    
    return(dataset)

# --- Architecture ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder - spatial reduction
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        
        self.bottleneck1 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=3, padding=1), nn.ReLU())
        
        # Decoder - spatial back to original
        self.dec3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1), nn.ReLU())
        
        # Time embedding
        self.time_embed = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)
                
    def forward(self, x, t):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        embed = self.time_embed(t.float().unsqueeze(1)).view(-1, 64, 1, 1)
        embed = embed.expand(-1, -1, e3.shape[2], e3.shape[3])
        
        e3 = torch.cat([e3, embed], dim=1)
        e3 = nn.LayerNorm(e3.shape[1:], elementwise_affine=False)(e3)
        
        b1 = self.bottleneck1(e3)
        
        d3 = self.dec3(b1)
        d2 = self.dec2(d3) + e1     # Skip con
        d1 = self.dec1(d2)
        
        return d1

# --- Forward Diffusion Process ---
def forward_diffusion(image, t):
    noise = torch.randn_like(image).to(device)

    root_image = alpha_cumulative[t].view(-1, 1, 1, 1).sqrt()       # Portion of original kept
    root_noise = (1 - alpha_cumulative[t]).view(-1, 1, 1, 1).sqrt() # Portion of added noise
    
    xt = root_image * image + root_noise * noise
    
    return xt, noise
    
# --- Sampling new images (reverse diffusion process) ---
def new_images(model):
    x = torch.randn((num_samples, 1, image_size, image_size), device=device)
    
    for t in reversed(range(T)):
        with torch.no_grad():
            t_tensor = torch.full((num_samples,), t, device=device)
            noise_pred = model(x, t_tensor)
            
            beta_t = beta[t]
            alpha_t = alpha[t]
            alpha_c = alpha_cumulative[t]
            
            if t > 0:
                noise = torch.randn_like(x).to(device)
            else:
                noise = torch.zeros_like(x).to(device)

            # Reverse diffusion step
            x = (1 / alpha_t.sqrt()) * (x - beta_t * noise_pred / (1 - alpha_c).sqrt()) + noise * beta_t.sqrt()

    return x.cpu()

# --- Hyperparameters ---
T = 1000                # Diffusion steps
beta_start = 1e-5       # Noise variance start
beta_end = 1e-2         # Final noise variance
image_size = 28
batch_size = 1
learning_rate = 1e-3
epochs = 350
num_samples = 5
losses = []

# --- Define noise-adding ---
beta = torch.linspace(beta_start, beta_end, T).to(device)   # Increasing coefficient of noise adding
alpha = 1 - beta                                            # Proportion of original image
alpha_cumulative = torch.cumprod(alpha, dim=0)              # Cumulative proportion of orig image

# --- Import dataset as list of arrays ---
dataset = images_to_list('dataset_ones').to(device)
data_loaded = DataLoader(dataset, batch_size, shuffle=True)

# --- Diffusion test loop ---
# sample = dataset[0].unsqueeze(0).to(device)

# fig, axes = plt.subplots(1, 10, figsize=(6, 15))
# for i in range(10):    
#     sample_diff, noise = forward_diffusion(sample, i * 99)
  
#     sample_diff = sample_diff.squeeze().cpu().numpy()
#     ax = axes[i]  # Select the i-th axis
#     ax.imshow(sample_diff, cmap='gray')
#     ax.axis('off')  # Remove axis for better visualization
    
# --- Main loop ---
for epoch in range(epochs):
    loss_c = 0
    
    for batch in data_loaded:
    
        batch_size_actual = batch.shape[0]
        t = torch.randint(0, T, (batch_size_actual,), device=device)
        xt, noise = forward_diffusion(batch, t)
        
        
# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()

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