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

# Get class information from data directories
digit_classes = sorted(os.listdir('./data/digits'))
operator_classes = sorted(os.listdir('./data/operators'))

class CombinedNet(nn.Module):
    def __init__(self):
        super(CombinedNet, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second conv block
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third conv block
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Shared features end here
        
        # Type classification (digit vs operator)
        self.type_fc = nn.Linear(128 * 4 * 4, 2)
        
        # Digit classification (0-9)
        self.digit_fc = nn.Linear(128 * 4 * 4, len(digit_classes))
        
        # Operator classification
        self.operator_fc = nn.Linear(128 * 4 * 4, len(operator_classes))

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
        
        # Flatten
        features = x.view(-1, 128 * 4 * 4)
        
        # Get all predictions
        type_out = self.type_fc(features)      # Is it digit or operator?
        digit_out = self.digit_fc(features)    # If digit, which one?
        operator_out = self.operator_fc(features)  # If operator, which one?
        
        return type_out, digit_out, operator_out

    def get_num_classes(self):
        """Return the number of classes for each head"""
        return {
            'type': 2,
            'digit': len(digit_classes),
            'operator': len(operator_classes)
        }

    @staticmethod
    def get_class_mappings():
        """Return the class mappings"""
        return {
            'digit': {i: c for i, c in enumerate(digit_classes)},
            'operator': {i: c for i, c in enumerate(operator_classes)}
        }

def print_model_info():
    """Print diagnostic information about the model"""
    print("\nDiagnostic Information:")
    print(f"Digit classes: {digit_classes}")
    print(f"Number of digit classes: {len(digit_classes)}")
    print(f"Operator classes: {operator_classes}")
    print(f"Number of operator classes: {len(operator_classes)}")
    
    print("\nNetwork Architecture:")
    print("CombinedNet outputs:")
    print(f"Type classification: 2 classes (digit/operator)")
    print(f"Digit classification: {len(digit_classes)} classes")
    print(f"Operator classification: {len(operator_classes)} classes")

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
    digit_net = CombinedNet().to(device)    # 10 for digits
    operator_net = CombinedNet().to(device)  # 9 for operators


    # Rest of your training code...
    # (Include all the training loop code here)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train_models()
    print_model_info()
