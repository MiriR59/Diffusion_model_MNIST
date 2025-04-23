import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms 
import torch.nn.init as init
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
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
    dataset = TensorDataset(image_tensor)

    return dataset, image_tensor

# --- Forward diffusion process --- 
def forward_diffusion(image, t):
    noise = torch.randn_like(image).to(device)

    root_image = alpha_cumulative[t].view(-1, 1, 1, 1).sqrt()       # Portion of original kept
    root_noise = (1 - alpha_cumulative[t]).view(-1, 1, 1, 1).sqrt() # Portion of added noise
    
    xt = root_image * image + root_noise * noise
    
    return xt, noise

# --- Reverse diffusion process (image reconstruction) ---
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
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t * noise_pred / torch.sqrt(1 - alpha_c))) + noise * torch.sqrt(beta_t)

    return x.cpu

# --- Encoder block ---
class Encoder_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1), nn.ReLU(),
                                     nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1), nn.ReLU())
        
        self.time_embed = nn.Sequential(nn.Linear(1, output_dim // 2), nn.ReLU(),
                                        nn.Linear(output_dim // 2, output_dim))
        
    def forward(self, x, t):
        x = self.encoder(x)
        t_proj = self.time_embed(t.float()).view(-1, self.output_dim, 1, 1)
        x = x + t_proj
        
        return x

# --- Bottleneck block ---
class Bottleneck_block(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.bottleneck = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1), nn.ReLU())
        
        self.time_embed = nn.Sequential(nn.Linear(1, input_dim // 2), nn.ReLU(),
                                        nn.Linear(input_dim // 2, input_dim))
    def forward(self, x, t):
        x = self.bottleneck(x)
        t_proj = self.time_embed(t.float()).view(-1, self.input_dim, 1, 1)
        x = x + t_proj
        
        return x

# --- Decoder block ---
class Decoder_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.decoder = nn.Sequential(nn.ConvTranspose2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                     nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1), nn.ReLU())
        
        self.time_embed = nn.Sequential(nn.Linear(1, output_dim // 2), nn.ReLU(),
                                        nn.Linear(output_dim // 2, output_dim))
        
    def forward(self, x, t):
        x = self.decoder(x)
        if self.output_dim > 1:
            t_proj = self.time_embed(t.float()).view(-1, self.output_dim, 1, 1)
            x = x + t_proj
        
        return x

# --- Architecture ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = Encoder_block(1, 32)
        self.enc2 = Encoder_block(32, 64)
        
        self.bottle1 = Bottleneck_block(64)
        self.bottle2 = Bottleneck_block(64)
        
        self.dec2 = Decoder_block(64, 32)
        self.dec1 = Decoder_block(32, 1)
        
        self.apply(self.init_weights)
    
    def init_weights(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            init.xavier_uniform_(layer.weight)
    
    def forward(self, x, t):
        e1 = self.enc1(x, t)
        e2 = self.enc2(e1, t)
        
        b1 = self.bottle1(e2, t)
        b2 = self.bottle2(b1)
        
        d2 = self.dec2(b2) + e1
        d1 = self.dec1(d2)
        
        return d1
    
# --- Hyperparameters ---
T = 2000
beta_start = 1e-5
beta_end = 2e-2
image_size = 28
batch_size = 25
learning_rate = 1e-4
epochs = 1501
num_samples = 1
losses = []

# --- Define noise-adding ---
beta = torch.linspace(beta_start, beta_end, T).to(device)   # Increasing coefficient of noise adding
alpha = 1 - beta                                            # Proportion of original image
alpha_cumulative = torch.cumprod(alpha, dim=0)              # Cumulative proportion of orig image

# --- Import dataset as list of arrays ---
dataset, image_tensor = images_to_list('dataset_ones')
data_loaded = DataLoader(dataset, batch_size, shuffle=True)

# --- Diffusion test loop ---
sample = image_tensor[0].unsqueeze(0).to(device)
fig, axes = plt.subplots(1, 10, figsize=(6, 15))
for i in range(10): 
     sample_diff, noise = forward_diffusion(sample, (((i + 1) * T - 1) // 10))
     sample_diff = sample_diff.squeeze().cpu().numpy()
     ax = axes[i]
     ax.imshow(sample_diff, cmap='gray')
     ax.axis('off')
# Adjust layout to avoid overlap
plt.tight_layout()
plt.show() 

# --- Model init ---
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.4)
loss_fn = nn.MSELoss() 

# --- Main loop ---
for epoch in range(epochs):
    loss_c = 0
    
    for batch in data_loaded:
        batch = batch[0].to(device)
        batch_size_actual = batch.shape[0]
        t = torch.randint(0, T, (batch_size_actual, 1), device=device)
        xt, noise = forward_diffusion(batch, t)
        
        noise_pred = model(xt, t)
        loss = loss_fn(noise_pred, noise)
        loss_c += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    loss_c /= batch_size
    losses.append(loss_c.item())
    print(f"Epoch {epoch+1}: Loss = {loss_c.item():.6f}, Learning Rate = {current_lr:.6e}")
    
    if (epoch + 1) % 250 == 0:

        # --- Test at multiple noise levels ---
        steps = [int(p * T) for p in [0.1, 0.25, 0.5, 0.75, 0.95]]
        fig, axs = plt.subplots(len(steps), 3, figsize=(10, 12))
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        img = Image.open('dataset_ones/1.png').convert('L')
        img = transform(img).unsqueeze(0).to(device)
        
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
            axs[i, 1].set_title(f"Noisy t={t}, alpha_c={alpha_cumulative[t].item():.4f}")

            axs[i, 2].imshow(denoised.squeeze().clamp(-1, 1).cpu(), cmap='gray')
            axs[i, 2].set_title(f"Step-by-step Denoised")

            for j in range(3):
                axs[i, j].axis('off')

        plt.tight_layout()
        plt.show()
    
        # Count how many parameter groups (with gradients) you want to visualize
        param_list = [p for p in model.parameters()]
        num_params = len(param_list)
        num_cols = 6
        num_rows = 5  # Calculate number of rows to fit all histograms
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        axes = axes.flatten()  # Flatten the axes array to iterate easily
        
        # Skip parameters from BatchNorm2d layers in Sequential modules
        param_idx = 0
        for name, param in model.named_parameters():
            if '.1.' in name:
                continue  # Skip BatchNorm or any module at index 1
            if param.grad is not None:
                axes[param_idx].hist(param.grad.cpu().numpy().flatten(), bins=100)
                axes[param_idx].set_title(f'Gradient Histogram - {name}')
                axes[param_idx].set_xlabel('Gradient Value')
                axes[param_idx].set_ylabel('Frequency')
            else:
                axes[param_idx].axis('off')
            param_idx += 1

        # Turn off any unused subplots
        for j in range(param_idx, len(axes)):
            axes[j].axis('off')
        
        fig.suptitle(f'Loss: {loss_c:.2e}, Epoch: {epoch + 1}, LR: {learning_rate:.2e}, CLR: {current_lr:.2e}', fontsize=30)
        plt.tight_layout()
        plt.show()


        
torch.save(model.state_dict(), "diffusion_model.pth")
print("Model saved successfully.")

# --- Plot loss curve ---
plt.plot(losses, label="Total loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss curve during training")
plt.legend()
plt.show()


