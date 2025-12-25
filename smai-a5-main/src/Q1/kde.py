# kde.py
import numpy as np

class CustomKDE:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        """
        Initialize the KDE model.
        """
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.data = None
        self.n = 0 # Number of training samples
        self.d = 0 # Dimensions (1 for gray, 3 for RGB)

    def fit(self, data, sample_ratio=1.0):
        """
        Stores the training data.
        Includes 'Smart Sampling' to reduce computational cost.
        """
        self.d = data.shape[1]
        total_samples = data.shape[0]
        
        # Smart Sampling: Randomly select a subset of the background pixels
        # If we use all pixels, prediction becomes too slow.
        if sample_ratio < 1.0:
            num_select = int(total_samples * sample_ratio)
            indices = np.random.choice(total_samples, num_select, replace=False)
            self.data = data[indices]
        else:
            self.data = data
            
        self.n = self.data.shape[0]
        print(f"KDE Fitted with {self.n} samples (Dimensions: {self.d})")

    def _kernel_function(self, u):
        """
        Applies the selected kernel function.
        u = (x - xi) / h
        """
        # We calculate based on squared distance to avoid expensive Sqrt operations where possible,
        # but strictly following the formula K(u):
        
        if self.kernel == 'gaussian':
            # K(u) = (1 / sqrt(2pi)) * exp(-0.5 * u^2)
            # Note: In multivariate, this is usually handled via covariance, 
            # but here we treat features as independent or use Euclidean norm.
            return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (u**2))
            
        elif self.kernel == 'triangular':
            # K(u) = 1 - |u| for |u| <= 1, else 0
            k = 1 - np.abs(u)
            return np.maximum(k, 0)
            
        elif self.kernel == 'uniform':
            # K(u) = 0.5 for |u| <= 1, else 0
            k = np.where(np.abs(u) <= 1, 0.5, 0)
            return k
            
        else:
            raise ValueError("Unknown kernel type")

    def predict(self, samples):
        """
        Predicts density for new samples.
        Formula: f(x) = (1 / (n * h^d)) * Sum( K( (x - xi) / h ) )
        """
        num_test = samples.shape[0]
        num_train = self.data.shape[0]
        
        # To avoid MemoryError with large matrices (Test x Train), 
        # we process the test samples in batches (chunks).
        batch_size = 100 
        densities = np.zeros(num_test)
        
        # Constant factor: 1 / (n * h^d)
        normalization_factor = 1.0 / (self.n * (self.bandwidth ** self.d))
        
        for i in range(0, num_test, batch_size):
            # Get a chunk of test pixels
            batch_samples = samples[i : i + batch_size]
            
            # Vectorized Distance Calculation using Broadcasting
            # Shape: (Batch_Size, 1, D) - (1, Train_Size, D) -> (Batch_Size, Train_Size, D)
            diff = batch_samples[:, np.newaxis, :] - self.data[np.newaxis, :, :]
            
            # Euclidean distance: sqrt(sum(diff^2))
            # We divide by bandwidth inside the norm or after. 
            # Formula says K((x-xi)/h). 
            dist = np.linalg.norm(diff, axis=2) # Shape: (Batch_Size, Train_Size)
            
            u = dist / self.bandwidth
            
            # Apply Kernel
            k_values = self._kernel_function(u)
            
            # Sum over all training samples (axis 1)
            sum_k = np.sum(k_values, axis=1)
            
            # Apply Normalization
            densities[i : i + batch_size] = normalization_factor * sum_k
            
        return densities