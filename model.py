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

# Define the network architectures
class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class OperatorNet(nn.Module):
    def __init__(self):
        super(OperatorNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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

    # Initialize networks
    digit_net = DigitNet().to(device)
    operator_net = OperatorNet().to(device)

    # Rest of your training code...
    # (Include all the training loop code here)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train_models()
