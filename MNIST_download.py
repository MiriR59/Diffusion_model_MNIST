import os
import torch
from torchvision import datasets, transforms
import torchvision
from PIL import Image

# Create transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download MNIST dataset (just once)
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

# Filter images where the label is 1
filtered_data = [(img, label) for img, label in dataset if label == 1]

# Select 200 images of the digit '1'
selected_data = filtered_data[:200]

# Ensure the directory exists to save images
os.makedirs('dataset_ones', exist_ok=True)

# Save the images
for idx, (img, label) in enumerate(selected_data):
    # Convert tensor back to PIL image and save as PNG
    img_pil = transforms.ToPILImage()(img)
    img_pil.save(f"dataset_ones/{idx}.png")

print("Saved 200 images of digit '1' to the folder 'one_digit_images'.")
