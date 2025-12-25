# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE

def get_dataloaders():
    """
    Loads Fashion-MNIST, applies normalization, and returns DataLoaders.
    """
    # Define transformations: Convert to Tensor and Normalize to [0, 1]
    # Note: ToTensor() converts [0, 255] -> [0.0, 1.0] automatically.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download and load training data
    train_dataset = datasets.FashionMNIST(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )

    # Download and load test data
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        transform=transform, 
        download=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    return train_loader, test_loader