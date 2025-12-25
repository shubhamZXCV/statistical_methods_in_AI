import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class MultiTaskFashionCNN(nn.Module):
    """
    A Convolutional Neural Network with a shared backbone and two heads:
    1. Classification (10 classes)
    2. Regression (1 scalar output)
    """
    def __init__(self, num_classes=10, dropout_rate=0.25):
        super().__init__()

        # Dictionary to store intermediate activations
        self.activations = {}
        
        # # --- 1. Shared Convolutional Backbone (Feature Extractor) ---
        # # Input image size: 1 x 28 x 28
        # self.features = nn.Sequential(
        #     # Block 1: Output size: 64 x 28 x 28 -> 64 x 14 x 14 (after MaxPool)
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2), 
        #     nn.Dropout(dropout_rate),

        #     # Block 2: Output size: 128 x 14 x 14 -> 128 x 7 x 7 (after MaxPool)
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(dropout_rate),

        #     # Block 3: Output size: 256 x 7 x 7 (No Pool to preserve features before flattening)
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     # Optional: Add one more MaxPool if 7x7 is too large, 
        #     # but 7*7*128 features is a good size.
        # )

        # --- 1. Shared Convolutional Backbone (Modified for easy access) ---
        # The 'features' sequential module is broken down into named blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout(dropout_rate)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # We need a list of modules to register hooks on
        self.feature_extractor_modules = {
            "block1": self.conv_block1,
            "block2": self.conv_block2,
            "block3": self.conv_block3,
        }
        
        # Calculate the size of the flattened feature vector: 128 channels * 7 * 7
        self.feature_dim = 128 * 7 * 7  

        # --- 2. Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes) # Outputs raw logits [B, 10]
        )
        
        # --- 3. Regression Head ---
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1) # Outputs scalar prediction [B, 1]
        )

    def register_hooks(self):
        """Register forward hooks to store the output of each convolutional block."""
        # Define the hook function
        def hook_fn(module, input, output, name):
            # Store the output tensor (activation map)
            self.activations[name] = output.detach() 

        # Register the hook for each block
        for name, module in self.feature_extractor_modules.items():
            # register_forward_hook takes a function (hook_fn) as input
            module.register_forward_hook(
                lambda module, input, output, name=name: hook_fn(module, input, output, name)
            )

    def forward(self, x):
        # Clear previous activations (important for batch processing)
        self.activations = {} 
        
        # 1. Shared Feature Extraction (Now using the named blocks)
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        shared_features = self.conv_block3(h)
        
        # Flatten the features for the fully connected layers
        flattened_features = shared_features.view(x.size(0), -1) 
        
        # 2. Classification Output
        logits = self.classifier(flattened_features) 
        
        # 3. Regression Output
        regression_prediction = self.regressor(flattened_features)
        
        # Return both outputs
        return logits, regression_prediction

    # # --- 2. Forward Pass ---
    # def forward(self, x):
    #     # 1. Shared Feature Extraction
    #     shared_features = self.features(x)
        
    #     # Flatten the features for the fully connected layers
    #     flattened_features = shared_features.view(x.size(0), -1) # x.size(0) is batch size
        
    #     # 2. Classification Output
    #     # logits: raw output before softmax
    #     logits = self.classifier(flattened_features) 
        
    #     # 3. Regression Output
    #     # prediction: scalar ink value
    #     regression_prediction = self.regressor(flattened_features)
        
    #     # Return both outputs
    #     return logits, regression_prediction


# Define the loss functions outside the training loop for efficiency
L_classification_fn = nn.CrossEntropyLoss() # L_CE
L_regression_fn = nn.MSELoss()               # L_MSE

def calculate_joint_loss(
    cls_logits, 
    cls_target, 
    reg_prediction, 
    reg_target, 
    lambda1, 
    lambda2
):
    """
    Calculates the total joint loss: L_total = lambda1*L_CE + lambda2*L_MSE
    
    Args:
        cls_logits (torch.Tensor): Model's classification output (logits, [B, 10]).
        cls_target (torch.Tensor): True class labels (integers, [B]).
        reg_prediction (torch.Tensor): Model's regression output (scalar, [B, 1]).
        reg_target (torch.Tensor): True ink targets (scalar, [B, 1]).
        lambda1 (float): Weight for the classification loss (L_CE).
        lambda2 (float): Weight for the regression loss (L_MSE).

    Returns:
        torch.Tensor: The total scalar joint loss.
    """
    
    # Ensure targets are on the correct device (if applicable) and type
    cls_target = cls_target.long() 
    reg_target = reg_target.float()
    
    # Calculate Classification Loss (Cross-Entropy)
    L_CE = L_classification_fn(cls_logits, cls_target)
    
    # Calculate Regression Loss (Mean Squared Error)
    L_MSE = L_regression_fn(reg_prediction, reg_target)
    
    # Calculate Joint Loss
    L_total = (lambda1 * L_CE) + (lambda2 * L_MSE)
    
    # For logging and analysis, return the individual losses as well
    return L_total, L_CE, L_MSE

def visualize_feature_maps(model, test_loader, num_images=3, num_maps_to_show=8):
    """
    Selects images from the test set, runs the forward pass, and plots feature maps.
    
    Args:
        model (MultiTaskFashionCNN): Trained model with registered hooks.
        test_loader (DataLoader): PyTorch DataLoader for the test set.
        num_images (int): Number of test images to visualize.
        num_maps_to_show (int): Number of feature channels to display per block.
    """
    # This is a safe way to ensure data goes where the model is.
    device = next(model.parameters()).device

    # Put model in evaluation mode
    model.eval()
    
    # Get a batch of images from the test set
    images, labels, ink_targets = next(iter(test_loader))

    # --- FIX: Move Images to the Device ---
    images = images.to(device)
    labels = labels.to(device) # Move labels too, though not strictly needed for this forward pass
    
    # Use only the first `num_images`
    images = images[:num_images]
    
    # Run forward pass (activations are stored via hooks)
    with torch.no_grad():
        model(images)
    
    print(f"\n--- Visualizing Feature Maps for {num_images} Images ---")
    
    for i in range(num_images):
        print(f"\nImage {i+1} (True Class: {labels[i].item()})")
        
        # 1. Plot the input image
        fig = plt.figure(figsize=(15, 6))
        
        # Prepare the input image for display (denormalize)
        # Assuming FASHION_MEAN=(0.2860), FASHION_STD=(0.3530)
        input_img = images[i].cpu().squeeze() * 0.3530 + 0.2860
        
        ax = fig.add_subplot(1, len(model.feature_extractor_modules) + 1, 1)
        ax.imshow(input_img.numpy(), cmap='gray')
        ax.set_title(f'Input Image (28x28)')
        ax.axis('off')
        
        # 2. Plot feature maps from each block
        for k, (name, act_tensor) in enumerate(model.activations.items()):
            # The activation tensor is (B, C, H, W). We take the i-th image.
            feature_maps = act_tensor[i].cpu().numpy() # Shape: (C, H, W)
            
            # Select and display the first N maps
            plot_maps = feature_maps[:num_maps_to_show]
            
            # Subplot layout: row 1: input, row 2-N: maps
            ax = fig.add_subplot(1, len(model.feature_extractor_modules) + 1, k + 2)
            
            # Montage the N maps into one image for visualization
            montage_rows = 2
            montage_cols = num_maps_to_show // montage_rows
            montage = np.zeros((plot_maps.shape[1] * montage_rows, plot_maps.shape[2] * montage_cols))
            
            for m in range(montage_rows):
                for n in range(montage_cols):
                    idx = m * montage_cols + n
                    if idx < num_maps_to_show:
                        montage[m*plot_maps.shape[1]:(m+1)*plot_maps.shape[1], 
                                n*plot_maps.shape[2]:(n+1)*plot_maps.shape[2]] = plot_maps[idx]
            
            # Display the montage
            ax.imshow(montage, cmap='viridis')
            ax.set_title(f'{name} ({feature_maps.shape[0]} Maps, {feature_maps.shape[1]}x{feature_maps.shape[2]})')
            ax.axis('off')

        plt.tight_layout()
        plt.show()