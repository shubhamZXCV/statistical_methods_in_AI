# NEURAL NETWORKS FROM SCRATCH

# -----------------------------
# Imports
# -----------------------------
import numpy as np
from typing import Tuple,Dict,List
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime # Added for better run naming
# import wandb

SEED = 42  
np.random.seed(SEED)
USERNAME = "shubham.goel"
# -----------------------------
# Activation function classes
# -----------------------------
class Activation:
    """This class is the base class for all activation classes"""
    def forward(self , X:np.ndarray)->np.ndarray:
        raise NotImplementedError
    def backward(self , X:np.ndarray)->np.ndarray:
        raise NotImplementedError

class ReLU(Activation):
    """ReLU activation function."""
    def forward(self, x):
        """ReLU forward: max(0, x)"""
        return np.maximum(0, x)
    def backward(self, x):
        """ReLU derivative: 1 if x > 0, else 0"""
        return (x > 0).astype(float)

class Tanh(Activation):
    """Tanh activation function."""
    def forward(self, x):
        """Tanh forward: (e^x - e^(-x)) / (e^x + e^(-x))"""
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return np.tanh(x)
    def backward(self, x):
        """Tanh derivative: 1 - tanh^2(x)"""
        tanh_x = self.forward(x)
        return 1 - tanh_x**2
    
class Sigmoid(Activation):
    """Sigmoid activation function."""
    def forward(self, x):
        """Sigmoid forward: 1 / (1 + e^(-x))"""
        return 1 / (1 + np.exp(-x))
    def backward(self, x):
        """Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))"""
        sig = self.forward(x)
        return sig * (1 - sig)

class Identity(Activation):
    """Identity activation function (no activation)."""
    def forward(self, x):
        """Identity forward: x"""
        return x
    def backward(self, x):
        """Identity derivative: 1"""
        return np.ones_like(x)
    
# -----------------------------
# Linear layer
# -----------------------------
class Linear:
    """
    Linear layer implementation with forward and backward passes.
    """
    def __init__(self, input_size, output_size, activation_function, learning_rate=0.5):
        """
        Initialize linear layer.
        
        Args:
            input_size (int): Number of input features
            output_size (int): Number of output features
            activation_function (ActivationFunction): Activation function to use
            learning_rate (float): Learning rate for parameter updates
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        
        # Set seed for reproducible initialization
        np.random.seed(42)
        # Xavier/Glorot initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))
        self.biases = np.zeros((1, output_size)) + 0.01
        # Storage for forward pass data
        self.last_input = None
        self.last_linear_output = None
        self.last_activation_output = None
        # Storage for cumulative gradients
        self.weight_gradients = np.zeros_like(self.weights)
        self.bias_gradients = np.zeros_like(self.biases)
        
    def forward(self, x):
        """
        Forward pass through the linear layer.
        
        Args:
            x (np.ndarray): Input data of shape (batch_size, input_size)
        """
        # Store input for backward pass
        self.last_input = x.copy()
        # Linear transformation: y = xW + b
        self.last_linear_output = np.dot(x, self.weights) + self.biases
        # Apply activation function
        self.last_activation_output = self.activation_function.forward(self.last_linear_output)
        return self.last_activation_output
    
    def backward(self, upstream_gradient):
        """
        Backward pass through the linear layer.
        
        Args:
            upstream_gradient (np.ndarray): Gradient from the next layer
            
        Returns:
            np.ndarray: Gradient to pass to the previous layer
        """
        batch_size = self.last_input.shape[0]
        # Gradient through activation function
        activation_gradient = self.activation_function.backward(self.last_linear_output)
        delta = upstream_gradient * activation_gradient # dL/da * da/dz
        # Gradient with respect to weights: dL/dW = X^T * delta basically dz/dw * dL/dz and z = X^TW + b
        weight_grad = np.dot(self.last_input.T, delta)
        # Gradient with respect to biases: dL/db = sum(delta, axis=0)
        bias_grad = np.sum(delta, axis=0, keepdims=True)
        # avg gradients for batch
        self.weight_gradients += weight_grad / batch_size
        self.bias_gradients += bias_grad / batch_size
        # Gradient with respect to input: dL/dX = delta * W^T
        input_gradient = np.dot(delta, self.weights.T)
        
        return input_gradient
    
    def update_parameters(self, gradient_accumulation_steps):
        """Update parameters using accumulated gradients."""
        self.weights -= (self.learning_rate / gradient_accumulation_steps) * self.weight_gradients 
        self.biases -= (self.learning_rate / gradient_accumulation_steps) * self.bias_gradients

    def zero_grad(self):
        """Reset accumulated gradients to zero."""
        self.weight_gradients.fill(0)
        self.bias_gradients.fill(0)
    
    def get_parameters(self):
        """Get current parameters."""
        return {
            'weights': self.weights.copy(),
            'biases': self.biases.copy()
        }
    
    def set_parameters(self, parameters):
        """Set parameters."""
        self.weights = parameters['weights'].copy()
        self.biases = parameters['biases'].copy()

# -----------------------------
# Losses
# -----------------------------

class Loss:
    """This is a base class for all loss classes"""
    def forward(self , predictions , targets)->np.ndarray:
        raise NotImplementedError
    def backward(self , predictions , targets)->np.ndarray:
        raise NotImplementedError
    
class MSELoss(Loss):
    """Mean Squared Error loss function."""
    def forward(self, predictions, targets):
        """MSE forward: 0.5 * mean((targets - predictions)^2)"""
        return 0.5 * np.mean((targets - predictions) ** 2)
    
    def backward(self, predictions, targets):
        """MSE backward: -1 * (targets - predictions) / batch_size"""
        batch_size = predictions.shape[0]
        return -1 * (targets - predictions) / batch_size

class BCELoss(Loss):
    """Binary Cross Entropy loss function."""
    def forward(self, predictions, targets):
        """BCE forward: -mean(targets * log(predictions) + (1-targets) * log(1-predictions))"""
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -1 * np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    def backward(self, predictions, targets):
        """BCE backward: Derivative w.r.t. the prediction p"""
        batch_size = predictions.shape[0]
        # Clipping is necessary to prevent division by exact zero
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -((targets / predictions) - ((1 - targets) / (1 - predictions))) / batch_size
    
# -----------------------------
# Model
# -----------------------------
class Model:
    """
    Neural network model class that manages layers and training.
    """
    def __init__(self, layers, loss_function):
        """
        Initialize the model.
        
        Args:
            layers (list): List of Linear layer objects
            loss_function (LossFunction): Loss function to use
        """
        self.layers = layers
        self.loss_function = loss_function
        self.training_history = {
            'losses': [],
            'samples_seen': []
        }
    
    def forward(self, x):
        """
        Forward pass through all layers.
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, loss_gradient):
        """
        Backward pass through all layers.
        """
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def train(self, x, y):
        """
        Perform one training step: forward pass, loss computation, and backpropagation.
        """
        # Forward pass
        predictions = self.forward(x)
        
        # Compute loss
        loss = self.loss_function.forward(predictions, y)
        
        # Backward pass
        loss_gradient = self.loss_function.backward(predictions, y)
        self.backward(loss_gradient)
        
        return loss
    
    def predict(self, x):
        """
        Make predictions on input data.
        """
        return self.forward(x)
    
    def zero_grad(self):
        """Reset gradients in all layers."""
        for layer in self.layers:
            layer.zero_grad()
    
    def update(self, gradient_accumulation_steps):
        """Update parameters in all layers and reset gradients."""
        for layer in self.layers:
            layer.update_parameters(gradient_accumulation_steps)
        self.zero_grad()
    
    def save_to(self, path):
        """
        Save model parameters to a file.
        """

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        # Collect all parameters
        parameters = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_parameters()
            parameters[f'layer_{i}_weights'] = layer_params['weights']
            parameters[f'layer_{i}_biases'] = layer_params['biases']
        
        np.savez(path, **parameters)
    
    def load_from(self, path):
        """
        Load model parameters from a file.
        """

        data = np.load(path)
        
        # Check if architecture matches
        for i, layer in enumerate(self.layers):
            weight_key = f'layer_{i}_weights'
            bias_key = f'layer_{i}_biases'
            
            if weight_key not in data or bias_key not in data:
                raise ValueError(f"Missing parameters for layer {i} in saved file")
            
            saved_weights = data[weight_key]
            saved_biases = data[bias_key]
            
            if (saved_weights.shape != layer.weights.shape or 
                saved_biases.shape != layer.biases.shape):
                raise ValueError(f"Parameter shape mismatch for layer {i}")


        # set the loaded parameters
        for i, layer in enumerate(self.layers):
            weight_key = f'layer_{i}_weights'
            bias_key = f'layer_{i}_biases'
            layer.set_parameters({
                'weights': data[weight_key],
                'biases': data[bias_key]
            })
    
    def get_parameter_count(self):
        """Get total number of parameters in the model."""
        total_params = 0
        for layer in self.layers:
            total_params += layer.weights.size + layer.biases.size
        return total_params



# -----------------------------
# Training loop utility
# -----------------------------

def train_model(model, dataset, batch_size=32, grad_accumulation_steps=1, 
                max_epochs=10000, patience=10, relative_loss_threshold=0.0001, 
                run_name=None, save_model=True, use_wandb = False):
    """
    Train a model with early stopping and progress tracking.
    
    Args:
        model (Model): Model to train
        dataset (BorderDataset): Dataset to train on
        batch_size (int): Size of each batch
        grad_accumulation_steps (int): Number of batches to accumulate gradients over
        max_epochs (int): Maximum number of epochs to train
        patience (int): Number of epochs to wait for improvement before stopping
        relative_loss_threshold (float): Relative improvement threshold for early stopping
        run_name (str): Name for this training run
        save_model (bool): Whether to save the model and plots
    
    Returns:
        dict: Training results including history and final metrics
    """
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create run directory
    run_dir = f"runs/{run_name}"
    if save_model:
        os.makedirs(run_dir, exist_ok=True)
    

    losses = []
    samples_seen = []
    epoch_losses = []
    total_samples = 0
    
    best_loss = float('inf')
    
    print(f"Starting training: {run_name}")
    
    
    # either trains till max_epochs or early stopping
    for epoch in range(max_epochs):
        epoch_loss = 0
        num_batches = 0

        # get the entire dataset
        all_coords, all_labels = dataset.get_all_samples()
        all_labels = all_labels.reshape(-1, 1)

        # make batches
        num_samples = len(all_coords)
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_coords = all_coords[batch_start:batch_end]
            batch_labels = all_labels[batch_start:batch_end]

            # training step
            batch_loss = model.train(batch_coords, batch_labels)
            epoch_loss += batch_loss
            num_batches += 1
            total_samples += len(batch_coords)

            # parameters will only be updated every grad_accumulation_steps
            if num_batches % grad_accumulation_steps == 0:
                model.update(grad_accumulation_steps)

            losses.append(batch_loss)
            samples_seen.append(total_samples)

            # Print batch progress (optional, can comment out if you want only epoch info)
            # print(f"Epoch {epoch+1}, Batch {num_batches}: Loss = {batch_loss:.6f}, Samples seen = {total_samples}")

        # Print epoch summary
        # avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        # if (epoch + 1) % 500 == 0 :
        #     print(f"Epoch {epoch+1} completed. Average Loss = {avg_epoch_loss:.6f}, Total Samples = {total_samples}")
        

        # final update if needed
        if num_batches % grad_accumulation_steps != 0:
            model.update(grad_accumulation_steps)
        

        # avg loss for this epoch
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        epoch_losses.append(avg_epoch_loss)
        

        # early stopping check
        if len(epoch_losses) > patience:
            recent_loss = avg_epoch_loss
            past_loss = epoch_losses[-(patience + 1)]
            
            # Stop immediately if loss hasn't improved by the threshold
            if recent_loss >= (1 - relative_loss_threshold) * past_loss:
                print(f"Early stopping at epoch {epoch}")
                break
        

        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}: Loss = {avg_epoch_loss:.6f}, Samples = {total_samples}")

    results = {
        'run_name': run_name,
        'final_loss': avg_epoch_loss,
        'total_epochs': epoch + 1,
        'total_samples': total_samples,
        'losses': losses,
        'samples_seen': samples_seen,
        'epoch_losses': epoch_losses,
        'model_parameters': model.get_parameter_count()
    }
    
    print(f"Training completed: {results['total_epochs']} epochs, {total_samples} samples")
    print(f"Final loss: {results['final_loss']:.6f}")
    
    if save_model:
        # save model params
        model.save_to(f"{run_dir}/model.npz")
        
        # save plot b/w training loss and no. of samples
        plt.figure(figsize=(10, 6))
        plt.plot(samples_seen, losses, alpha=0.7)
        plt.xlabel('Number of Samples Seen')
        plt.ylabel('Training Loss')
        plt.title('Training Loss vs Number of Samples Seen')
        plt.grid(True)
        plt.text(0.95, 0.95, USERNAME, ha='right', va='top', transform=plt.gca().transAxes, fontsize=10, color='gray', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{run_dir}/training_loss.png", dpi=150)
        plt.show()
        
        print(f"Results saved to {run_dir}/")
        
    else: 
        plt.figure(figsize=(10, 6))
        plt.plot(samples_seen, losses, alpha=0.7)
        plt.xlabel('Number of Samples Seen')
        plt.ylabel('Training Loss')
        plt.title('Training Loss vs Number of Samples Seen')
        plt.grid(True)
        plt.text(0.95, 0.95, USERNAME, ha='right', va='top', transform=plt.gca().transAxes, fontsize=10, color='gray', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    return results