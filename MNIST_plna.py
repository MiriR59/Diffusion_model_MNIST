import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms 
import torch.nn.init as init
import torch.nn.functional as F
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
            
            return img  # Return as NumPy array

    # Define transformation (ToTensor and Normalize)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the dataset (images from the 'image_folder')
    dataset = CustomImageDataset(image_folder=image_folder, transform=transform)

    # Get list of NumPy arrays representing the images
    image_list = [dataset[i] for i in range(len(dataset))]

    # Convert list of arrays to a 4D tensor
    image_tensor = torch.tensor(np.array(image_list)).float()
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
            t_tensor = torch.full((num_samples, 1), t, device=device)
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

    return x.cpu()

# --- Model performance test for different noise levels ---
def test_noise(num_samples=1, image_path='dataset_ones/1.png'):
    
    # --- Define noise levels as percentages of T ---
    steps = [int(p * T) for p in [0.1, 0.25, 0.5, 0.75, 0.95]]
    fig, axs = plt.subplots(len(steps), 3, figsize=(10, 12))

    # --- Load and prepare the image ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    img = Image.open(image_path).convert('L')
    img = transform(img).unsqueeze(0).to(device)

    # --- Loop through selected noise levels ---
    for i, t in enumerate(steps):
        t_tensor = torch.tensor([t], device=device)
        xt, true_noise = forward_diffusion(img, t_tensor)

        # --- Reverse diffusion ---
        x = xt.clone()
        for rev_t in reversed(range(t + 1)):
            with torch.no_grad():
                t_rev = torch.full((num_samples, 1), rev_t, device=device)
                noise_pred = model(x, t_rev)

                beta_t = beta[rev_t]
                alpha_t = alpha[rev_t]
                alpha_c = alpha_cumulative[rev_t]

                noise = torch.randn_like(x) if rev_t > 0 else torch.zeros_like(x)
                x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t * noise_pred / torch.sqrt(1 - alpha_c))) + noise * torch.sqrt(beta_t)

        denoised = x

        # --- Plot results ---
        axs[i, 0].imshow(img.squeeze().cpu(), cmap='gray')
        axs[i, 0].set_title("Original")

        axs[i, 1].imshow(xt.squeeze().cpu(), cmap='gray')
        axs[i, 1].set_title(f"Noisy t={t}, αᶜ={alpha_cumulative[t].item():.4f}")

        axs[i, 2].imshow(denoised.squeeze().clamp(-1, 1).cpu(), cmap='gray')
        axs[i, 2].set_title("Step-by-step Denoised")

        for j in range(3):
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

# --- Plotting histograms of gradients ---
def gradients_histogram():
    # Get all parameters with gradients and skip BatchNorm if needed
    param_items = [(name, p) for name, p in model.named_parameters() if p.grad is not None and '.1.' not in name]
    num_params = len(param_items)

    # Dynamically choose grid size
    num_cols = 6
    num_rows = math.ceil(num_params / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    axes = axes.flatten()

    for idx, (name, param) in enumerate(param_items):
        axes[idx].hist(param.grad.cpu().numpy().flatten(), bins=100)
        axes[idx].set_title(f'Gradient Histogram - {name}')
        axes[idx].set_xlabel('Gradient Value')
        axes[idx].set_ylabel('Frequency')

    # Hide any unused axes
    for j in range(len(param_items), len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Loss: {loss_c:.2e}, Epoch: {epoch + 1}, LR: {learning_rate:.2e}, CLR: {current_lr:.2e}', fontsize=30)
    plt.tight_layout()
    plt.show()

class Encoder_block(nn.Module):
    'pad2: Yes or No, specifies wheter to pad to the next power of 2'
    def __init__(self, input_dim, output_dim, stride=2, pad2='No'):
        super().__init__()
        self.input_dim = input_dim
        self.stride = stride
        self.pad2 = pad2
        self.encoder = nn.Sequential(
                                     nn.Conv2d(input_dim + output_dim // 2, input_dim, kernel_size=3, padding=1),
                                     nn.GroupNorm(num_groups=8, num_channels=input_dim),
                                     nn.SiLU(),
                                     nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=self.stride, padding=1),
                                     nn.GroupNorm(num_groups=8, num_channels=output_dim),
                                     nn.SiLU()
                                     )
        
        self.time_embed = nn.Sequential(
                                        nn.Linear(1, output_dim // 2),
                                        nn.SiLU(),
                                        nn.Linear(output_dim // 2, output_dim // 2)
                                        )
        
    def padding_2(self, x):
        height = x.size(2)
        width = x.size(3)
        
        check_height = 2 ** (height - 1).bit_length()
        check_width = 2 ** (width - 1).bit_length()
        
        if height != check_height or width != check_width:
            self.pad_height = (check_height - height) // 2
            self.pad_width = (check_width - width) // 2
            
            x = F.pad(x, (self.pad_width, self.pad_width, self.pad_height, self.pad_height))
            
        return x
    
    def forward(self, x, t):
        if self.pad2 == 'Yes':
            x = self.padding_2(x)
        
        B, _, H, W = x.shape
        t_proj = self.time_embed(t.float()).view(B, -1, 1, 1).expand(-1, -1, H, W)
           
        x = torch.cat([x, t_proj], dim=1)
        x = self.encoder(x)
        
        return x

class Decoder_block(nn.Module):
    def __init__(self, input_dim, output_dim, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.output_padding = output_padding
        self.decoder = nn.Sequential(
                                     nn.ConvTranspose2d(input_dim + input_dim // 2, input_dim, kernel_size=3, stride=self.stride, padding=self.padding, output_padding=self.output_padding),
                                     nn.GroupNorm(num_groups=8, num_channels=input_dim),
                                     nn.SiLU(),
                                     nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
                                     nn.GroupNorm(num_groups=8, num_channels=output_dim),
                                     nn.SiLU()
                                     )
        
        self.time_embed = nn.Sequential(
                                        nn.Linear(1, input_dim // 2),
                                        nn.SiLU(),
                                        nn.Linear(input_dim // 2, input_dim // 2)
                                        )
        
    def forward(self, x, t):
        B, _, H, W = x.shape
        t_proj = self.time_embed(t.float()).view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        x = torch.cat([x, t_proj], dim=1)
        x = self.decoder(x)
        return x

class ResNet_block(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.resnet = nn.Sequential(
                                    nn.Conv2d(input_dim + input_dim // 2, input_dim, kernel_size=3, padding=1),
                                    nn.GroupNorm(num_groups=8, num_channels=input_dim),
                                    nn.SiLU(),
                                    nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
                                    nn.GroupNorm(num_groups=8, num_channels=input_dim),  
                                    nn.SiLU()
                                    )
        
        self.time_embed = nn.Sequential(
                                        nn.Linear(1, input_dim // 2),
                                        nn.SiLU(),
                                        nn.Linear(input_dim // 2, input_dim // 2)
                                        )
    def forward(self, x, t):
        B, _, H, W = x.shape
        t_proj = self.time_embed(t.float()).view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        out = torch.cat([x, t_proj], dim=1)
        
        return x + self.resnet(out)

class Attention_block(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=4, num_channels=input_dim)
        self.Q = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.K = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.V = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.output = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        norm = self.norm(x)
        Q = self.Q(norm).reshape(B, C, -1)
        K = self.K(norm).reshape(B, C, -1)
        V = self.V(norm).reshape(B, C, -1)
        
        out = torch.bmm(Q.transpose(1,2), K) / (C ** 0.5)
        out = torch.softmax(out, dim=-1)
        out = torch.bmm(V, out.transpose(1,2)).reshape(B, C, H, W)
        
        return x + self.output(out)

# --- Architecture ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.SiLU())
        self.enc0 = Encoder_block(32, 64, stride=1, pad2='Yes')
        self.res1 = ResNet_block(64)
        self.enc1 = Encoder_block(64, 128)    
        self.res2 = ResNet_block(128)
        self.att1 = Attention_block(128)
        self.enc2 = Encoder_block(128, 128)
        self.res3 = ResNet_block(128)
        self.enc3 = Encoder_block(128, 256)
        
        self.res4 = ResNet_block(256)
        self.att2 = Attention_block(256)
        self.res5 = ResNet_block(256)
        
        self.dec4 = Decoder_block(256, 128)
        self.res6 = ResNet_block(128)
        self.dec3 = Decoder_block(128, 128)
        self.att3 = Attention_block(128)
        self.res7 = ResNet_block(128)
        self.dec2 = Decoder_block(128, 64)
        self.res8 = ResNet_block(64)
        self.dec1 = Decoder_block(64, 32, stride=1, padding=3, output_padding=0)
        self.output = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        self.apply(self.init_weights)
    
    def init_weights(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            init.kaiming_normal_(layer.weight, nonlinearity='relu')
    
    def forward(self, x, t):
        en = self.entry(x)
        x0 = self.enc0(en, t)
        x1 = self.res1(x0, t)
        x2 = self.enc1(x1, t)
        x3 = self.res2(x2, t)
        x4 = self.att1(x3)
        x5 = self.enc2(x4, t)
        x6 = self.res3(x5, t)
        x7 = self.enc3(x6, t)
        
        x8 = self.res4(x7, t)
        x9 = self.att2(x8)
        x10 = self.res5(x9, t)
        
        x11 = self.dec4(x10, t) + x6
        x12 = self.res6(x11, t) + x5
        x13 = self.dec3(x12, t)
        x14 = self.att3(x13)
        x15 = self.res7(x14, t)
        x16 = self.dec2(x15, t)
        x17 = self.res8(x16, t) + x1
        x18 = self.dec1(x17, t)
        out = self.output(x18)
        
        return out
    
# --- Hyperparameters ---
T = 500
beta_start = 5e-4
beta_end = 5e-2
image_size = 28
batch_size = 200
learning_rate = 1e-3
epochs = 5000
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

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma = 0.1)
loss_fn = nn.MSELoss() 

# --- Main loop ---
for epoch in range(epochs):
    loss_c = 0
    
    for batch in data_loaded:
        batch = batch[0].to(device)
        batch_size_actual = batch.shape[0]
        t = torch.randint(0, T, (batch_size_actual, 1), device=device)
        t = t // (T - 1)
        xt, noise = forward_diffusion(batch, t)
        noise_pred = model(xt, t)
        
        # t_float = t.float()
        # t_float.requires_grad_()
        # noise_pred = model(xt, t_float)
        # noise_pred.mean().backward()
        # print(t_float.grad)
        
        loss = loss_fn(noise, noise_pred)
        loss_c += loss

        optimizer.zero_grad()
        loss.backward()
        
        # # Vanishing gradient control
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: grad std = {param.grad.std().item():.2e}")
        mean = noise_pred.mean().item()
        std = noise_pred.std().item()
        print(f"mean noise_pred: {mean:.6f}")
        print(f"std noise_pred: {std:.6f}")


        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    loss_c /= (200 / batch_size)
    losses.append(loss_c.item())
    print(f"Epoch {epoch+1}: Loss = {loss_c.item():.6f}, Learning Rate = {current_lr:.6e}")
    if (epoch + 1) % 10 == 0:
        # Optional: log pre-clipping norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Pre-clip grad norm: {total_norm:.4f}")
        
    if (epoch + 1) % 200 == 0:
        # ew_images(model)
        model.eval()       
        test_noise()
        model.train()
        # gradients_histogram()

torch.save(model.state_dict(), "diffusion_model.pth")
print("Model saved successfully.")

# --- Plot loss curve ---
plt.plot(losses, label="Total loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss curve during training")
plt.legend()
plt.show()


