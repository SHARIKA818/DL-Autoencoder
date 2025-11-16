# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## THEORY
A Convolutional Autoencoder (CAE) for image denoising is a deep learning model designed to remove noise from corrupted images and restore them to their original quality. It consists of two main parts — an encoder, which compresses the input image into a lower-dimensional latent representation, and a decoder, which reconstructs the clean image from this representation. Convolutional layers are used to capture spatial and structural features, making the model efficient for image-related tasks. During training, the CAE learns to minimize the difference between noisy and clean images using a loss function such as Mean Squared Error (MSE). Once trained, it can effectively filter out unwanted noise while preserving key image details and textures. This makes it highly useful in applications like medical imaging, photography, and computer vision preprocessing.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Load PyTorch, torchvision, matplotlib, and other required modules.

### STEP 2: 

Use MNIST dataset and apply transformations (tensor conversion, normalization).
### STEP 3: 

Introduce random noise to input images to train the model for denoising.
### STEP 4: 

Define the Denoising Autoencoder with encoder–decoder convolutional layers.
### STEP 5: 

Use MSE loss and Adam optimizer to minimize reconstruction error for several epochs.
### STEP 6: 

Display original, noisy, and denoised images to evaluate performance visually.

## PROGRAM

### Name: Sharika R

### Register Number: 212223230204

```python
# Autoencoder for Image Denoising using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
     x=self.encoder(x)
     x=self.decoder(x)
     return x

# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Print model summary
print("Name: Sharika R")
print("Register Number: 212223230204")
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
  model.train()
  print("Name: Sharika R")
  print("Register Number: 212223230204")
  for epoch in range(epochs):
    running_loss = 0.0
    for images, _ in loader:
        images = images.to(device)
        noisy_images = add_noise(images).to(device)

        outputs = model(noisy_images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader)}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

   print("Name: Sharika R")
   print("Register Number: 212223230204")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)
```

### OUTPUT

### Model Summary
<img width="573" height="454" alt="image" src="https://github.com/user-attachments/assets/1bdd27b1-ff30-47df-98fb-bb844b1f6a0c" />

### Training loss
<img width="331" height="147" alt="image" src="https://github.com/user-attachments/assets/530dfa6f-3944-4c89-a1a7-b00dc52cdc6d" />

## Original vs Noisy Vs Reconstructed Image
<img width="1687" height="591" alt="image" src="https://github.com/user-attachments/assets/9f206937-9cb0-4f69-9364-e534660e4bf8" />

## RESULT
Thus the program for Image Denoising using Convolutional Autoencoder is implemented successfully.
