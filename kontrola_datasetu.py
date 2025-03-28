import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

# --- Dataset Preparation ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)

# Filter for digit '5' only
filtered_data = [(img, label) for img, label in dataset if label == 1]

# Create dataset of '5'
dataset_five = torch.utils.data.TensorDataset(
    torch.stack([img for img, _ in filtered_data]), 
    torch.tensor([label for _, label in filtered_data])
)

print(f"Number of images of '1': {len(dataset_five)}")

# --- Display Random 50 Samples ---
num_samples = min(50, len(dataset_five))
indices = random.sample(range(len(dataset_five)), num_samples)

fig, axes = plt.subplots(5, 10, figsize=(15, 7))
fig.suptitle("Random 50 Samples of the Digit '1' from MNIST", fontsize=16)

for idx, ax in zip(indices, axes.flatten()):
    img, label = dataset_five[idx]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.axis("off")

plt.tight_layout()
plt.show()
