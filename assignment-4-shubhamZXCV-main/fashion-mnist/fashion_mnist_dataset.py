# imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np

# --- Global Constants for Fashion-MNIST ---
# Fashion-MNIST statistics (often used for normalization)
FASHION_MEAN = (0.2860,)
FASHION_STD = (0.3530,)
# Image size
IMAGE_SIZE = 28


# --- 1. Custom Dataset Class (FashionMNISTDataset) ---
class FashionMNISTDataset(Dataset):
    """
    Custom Dataset class for Fashion-MNIST returning (image, class_label, ink_target).
    The 'ink_target' is the normalized mean pixel intensity of the image.
    """
    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset (torchvision.datasets.FashionMNIST): The base FashionMNIST dataset object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = dataset
        self.transform = transform
        
       # --- FIX STARTS HERE ---
        
        # 1. Determine the raw image data for this specific dataset/subset
        if isinstance(self.dataset, Subset):
            # For a Subset, we access the data via the base dataset and the indices
            base_dataset = self.dataset.dataset 
            indices = self.dataset.indices
            # Select the images corresponding to this subset's indices
            raw_images_tensor = base_dataset.data[indices]
        else:
            # For the base test dataset, access data directly
            raw_images_tensor = self.dataset.data

        # 2. Normalize raw pixels to [0, 1] for ink calculation
        raw_images = raw_images_tensor.float() / 255.0

        # 3. Calculate the average pixel value (ink target)
        # Shape: N x 1
        self.ink_targets_raw = torch.mean(raw_images, dim=(1, 2)) 
        
        # 4. Global normalization (Note: In a real multi-task setting, 
        # you might want to calculate mean/std over the *entire* # original training set for consistent normalization, but for 
        # this assignment, calculating it per split is fine if not specified otherwise.)
        self.ink_mean = torch.mean(self.ink_targets_raw)
        self.ink_std = torch.std(self.ink_targets_raw)
        
        self.ink_targets_normalized = (self.ink_targets_raw - self.ink_mean) / self.ink_std
        
        # --- FIX ENDS HERE ---

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Get image and class label from the base dataset
        image, class_label = self.dataset[idx]
        
        # 2. Get ink target
        # The ink target is a 0-dimensional tensor (a single float), 
        # so we unsqueeze it to make it a (1,) tensor for consistency 
        # when dealing with loss functions and models expecting a tensor.
        ink_target = self.ink_targets_normalized[idx].unsqueeze(0) 

        # 3. Apply transformations (including image normalization)
        if self.transform:
            image = self.transform(image)
        
        # Return tuple (image, class label, ink target)
        return image, class_label, ink_target


# --- 2. Data Augmentation Setup (Transforms) ---

# 3. Augmentations: Light augmentations for the training set only
# Note: transforms.ToTensor() converts the PIL image (0-255) to a tensor (0.0-1.0)
# transforms.Normalize then scales it using the Fashion-MNIST mean/std.

# Training Augmentations
train_transform = transforms.Compose([
    # Convert PIL Image to Tensor
    transforms.ToTensor(), 
    # Light Augmentations
    # Randomly crop and pad slightly 
    transforms.RandomCrop(IMAGE_SIZE, padding=4), 
    # Small rotation
    transforms.RandomRotation(degrees=5), 
    # Normalize the pixel values (Important for CNN performance)
    transforms.Normalize(FASHION_MEAN, FASHION_STD) 
])

# Validation and Test Transforms (Unaltered, only ToTensor and Normalize)
val_test_transform = transforms.Compose([
    # Convert PIL Image to Tensor
    transforms.ToTensor(), 
    # Normalize the pixel values
    transforms.Normalize(FASHION_MEAN, FASHION_STD)
])

# --- 3. Dataset Loading Function (load_fashion_data) ---

def load_fashion_data(data_dir='./data', val_split_ratio=0.1, batch_size=64):
    """
    Loads Fashion-MNIST, splits training data into train/val, wraps in custom dataset,
    applies transforms, and returns DataLoaders.

    Args:
        data_dir (str): Directory to save/load data.
        val_split_ratio (float): Fraction of the training data to use for validation (e.g., 0.1 for 90/10).
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # 1. Load the entire official train and test sets
    
    # Base Training Dataset (used for train/val split) - NO transform applied initially
    # download=True ensures it's available
    full_train_dataset_base = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=None
    )

    # Base Test Dataset (used for final evaluation) - NO transform applied initially
    test_dataset_base = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=None
    )
    
    # 2. Split the original training set into train and val (ensuring no overlap)
    
    total_train_size = len(full_train_dataset_base)
    val_size = int(val_split_ratio * total_train_size)
    train_size = total_train_size - val_size
    
    # Use random_split to create indices for a clean split
    train_indices, val_indices = random_split(
        range(total_train_size), 
        lengths=[train_size, val_size], 
        generator=torch.Generator().manual_seed(42) # Set seed for reproducibility
    )
    
    # Create Subset objects using the base dataset and the indices
    # This ensures a clean separation (no data leakage)
    train_subset = Subset(full_train_dataset_base, train_indices)
    val_subset = Subset(full_train_dataset_base, val_indices)

    # 3. Create Custom Datasets with appropriate transforms
    # Pass the subsets/base datasets to the custom wrapper
    train_dataset = FashionMNISTDataset(dataset=train_subset, transform=train_transform)
    val_dataset = FashionMNISTDataset(dataset=val_subset, transform=val_test_transform)
    test_dataset = FashionMNISTDataset(dataset=test_dataset_base, transform=val_test_transform)
    
    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    
    print(f"Total Train Size: {total_train_size}")
    print(f"Train/Val Split: {len(train_dataset)} / {len(val_dataset)}")
    print(f"Test Size: {len(test_dataset)}")
    print("Data Loaders created successfully.")
    
    return train_loader, val_loader, test_loader