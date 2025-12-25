# config.py

# Image Processing Parameters
IMG_SIZE = (100, 100)  # Resizing to (100, 100) to balance quality vs speed (N^2 complexity)
FEATURE_TYPE = 'rgb'   # 'rgb' or 'grayscale'

# KDE Model Parameters
KERNEL_TYPE = 'gaussian'  # Options: 'gaussian', 'triangular', 'uniform'
BANDWIDTH = 0.2           # Controls smoothness. Lower = more sensitive, Higher = smoother
SAMPLE_RATIO = 0.1        # Smart Sampling: Use only 10% of background pixels for training to save speed
THRESHOLD = 30      # Probability threshold. Below this = Foreground

# File Paths (Change these to your actual image paths)
BG_IMAGE_PATH = '../../dataset/Q1/back.jpg'
TEST_IMAGE_PATH = '../../dataset/Q1/Full.jpg'