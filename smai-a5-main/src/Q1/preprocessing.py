# preprocessing.py
import cv2
import numpy as np

def load_and_preprocess(path, target_size, feature_type='rgb'):
    """
    Loads an image, resizes it, and extracts features.
    
    Args:
        path (str): Path to image file.
        target_size (tuple): (width, height).
        feature_type (str): 'rgb' or 'grayscale'.
        
    Returns:
        original_img (np.array): The resized image for visualization.
        features (np.array): Flattened feature matrix (N_pixels, n_features).
    """
    # Load image
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not find image at {path}")
    
    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Image Alignment and Resizing
    # Resize to reduce computational cost of KDE
    img_resized = cv2.resize(img, target_size)
    
    # 2. Feature Extraction
    if feature_type == 'grayscale':
        # Convert to grayscale (1 channel)
        img_processed = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        # Flatten: (H, W) -> (N, 1)
        features = img_processed.reshape(-1, 1)
    else:
        # Use RGB channels (3 channels)
        img_processed = img_resized
        # Flatten: (H, W, 3) -> (N, 3)
        features = img_processed.reshape(-1, 3)
        
    # Normalize features to [0, 1] range. 
    # This is crucial for KDE bandwidth to work consistently.
    features = features.astype(np.float64) / 255.0
    
    return img_resized, features