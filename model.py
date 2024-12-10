import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau



class ImprovedNet(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedNet, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 32x32 -> 32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)  # 32x32 -> 32x32
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Second conv block
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 16x16 -> 16x16
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)  # 16x16 -> 16x16
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Third conv block
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)  # 8x8 -> 8x8
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        
        # Flatten and fully connected
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    


def train_models():
    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create datasets
    digit_dataset = torchvision.datasets.ImageFolder(
        root='./data/digits',
        transform=transform
    )
    operator_dataset = torchvision.datasets.ImageFolder(
        root='./data/operators',
        transform=transform
    )

    # Create dataloaders
    digit_trainloader = torch.utils.data.DataLoader(
        digit_dataset, 
        batch_size=32,
        shuffle=True, 
        num_workers=2
    )
    operator_trainloader = torch.utils.data.DataLoader(
        operator_dataset, 
        batch_size=32,
        shuffle=True, 
        num_workers=2
    )

    # Initialize the improved networks
    digit_net = ImprovedNet(num_classes=10).to(device)    # 10 for digits
    operator_net = ImprovedNet(num_classes=9).to(device)  # 9 for operators


    # Rest of your training code...
    # (Include all the training loop code here)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train_models()
