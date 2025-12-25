import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

SEED = 42  
np.random.seed(SEED)
USERNAME = "shubham.goel"

class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input
    
class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)
    
class Sigmoid:
    def forward(self, x):
        clipped = np.clip(x, -60, 60)
        self.output = 1 / (1 + np.exp(-clipped))
        return self.output
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)
    
class Identity:
    def forward(self, x):
        return x
    def backward(self, grad_output):
        return grad_output

class Linear:
    def __init__(self, din, dout, act):
        self.din = din
        self.dout = dout
        self.act = act
        limit = np.sqrt(6.0 / (din + dout))
        self.W = np.random.uniform(-limit, limit, size=(din, dout)).astype(np.float32)
        self.b = np.zeros((1, dout), dtype=np.float32)
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        self.y = self.act.forward(self.z)
        return self.y

    def backward(self, gy):
        gz = self.act.backward(gy)
        batch_size = self.x.shape[0]
        self.gW += self.x.T @ gz / batch_size
        self.gb += np.sum(gz, axis=0, keepdims=True) / batch_size
        return gz @ self.W.T

    def zero_grad(self):
        self.gW.fill(0)
        self.gb.fill(0)

    def step(self, lr):
        self.W -= lr * self.gW
        self.b -= lr * self.gb

class MLPAutoencoder:
    def __init__(self, enc_dims, bottleneck, dec_dims, act_type='relu'):
        """
        enc_dims: list of encoder hidden layer sizes
        bottleneck: latent dimension size
        dec_dims: list of decoder hidden layer sizes
        act_type: 'relu', 'tanh', or 'sigmoid'
        """
        self.layers = []
        
        # Helper to create new activation instance (each layer needs its own)
        def make_act():
            if act_type.lower() == 'relu':
                return ReLU()
            elif act_type.lower() == 'tanh':
                return Tanh()
            elif act_type.lower() == 'sigmoid':
                return Sigmoid()
            else:
                return ReLU()
        
        # Encoder
        prev = enc_dims[0]
        for d in enc_dims[1:]:
            self.layers.append(Linear(prev, d, make_act()))
            prev = d
        self.layers.append(Linear(prev, bottleneck, make_act()))  # bottleneck
        
        # Decoder
        prev = bottleneck
        for d in dec_dims:
            self.layers.append(Linear(prev, d, make_act()))
            prev = d
        self.layers.append(Linear(prev, enc_dims[0], Sigmoid()))  # output [0,1]
    
    def forward(self, x):
        h = x
        for lyr in self.layers:
            h = lyr.forward(h)
        return h
    
    def backward(self, grad):
        g = grad
        for lyr in reversed(self.layers):
            g = lyr.backward(g)
        return g
    
    def zero_grad(self):
        for lyr in self.layers:
            lyr.zero_grad()
    
    def update(self, lr):
        for lyr in self.layers:
            lyr.step(lr)
        self.zero_grad()
    
    def mse_loss(self, y, yhat):
        """MSE loss and gradient"""
        y = y.reshape(yhat.shape)
        err = yhat - y
        loss = float(np.mean(err ** 2))
        grad = (2 * err) / yhat.shape[0]  # divide by batch size, not total size
        return loss, grad
    
    def train_step(self, x, y):
        """Single training step"""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        yhat = self.forward(x)
        loss, grad = self.mse_loss(y, yhat)
        self.backward(grad)
        return loss
    
    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.forward(x)
    
    def save(self, path):
        data = {}
        for i, lyr in enumerate(self.layers):
            data[f'w{i}'] = lyr.W
            data[f'b{i}'] = lyr.b
        np.savez(path, **data)
    
    def load(self, path):
        data = np.load(path)
        for i, lyr in enumerate(self.layers):
            lyr.W = data[f'w{i}']
            lyr.b = data[f'b{i}']

def train_autoencoder(model, X_train, epochs=10, batch_size=128, lr=0.01,
                     show_progress=False, verbose=True, log_every=1):
    """Train autoencoder with minibatch SGD."""
    n = len(X_train)
    losses = []

    for epoch in range(epochs):
        idx = np.random.permutation(n)
        X_shuf = X_train[idx]

        epoch_loss = 0.0
        iterator = range(0, n, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f'Epoch {epoch+1}/{epochs}', leave=False)

        for step, i in enumerate(iterator):
            batch = X_shuf[i:i+batch_size]
            loss = model.train_step(batch, batch)  # autoencoder: input = target
            model.update(lr)
            epoch_loss += loss * len(batch)
            if show_progress:
                iterator.set_postfix({'loss': f'{loss:.6f}'})

        avg_loss = epoch_loss / n
        losses.append(avg_loss)

        if verbose and (epoch == 0 or epoch == epochs - 1 or (epoch + 1) % log_every == 0):
            print(f'Epoch {epoch+1} avg loss: {avg_loss:.6f}')

    return losses

def train_lfw_autoencoder(bottleneck_dim, X_normal, epochs=80, batch_size=64, lr=0.01,
                           act_type='tanh', show_progress=False, verbose=False, log_every=5):
    """Train autoencoder on z-scored normal faces and return fitted scaler."""
    input_dim = X_normal.shape[1]
    mean = X_normal.mean(axis=0, keepdims=True).astype(np.float32)
    std = X_normal.std(axis=0, keepdims=True).astype(np.float32) + 1e-6
    X_proc = ((X_normal - mean) / std).astype(np.float32)

    model = MLPAutoencoder(
        enc_dims=[input_dim, 1024, 512],
        bottleneck=bottleneck_dim,
        dec_dims=[512, 1024],
        act_type=act_type
    )

    losses = train_autoencoder(
        model,
        X_proc,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        show_progress=show_progress,
        verbose=verbose,
        log_every=log_every
    )
    return model, losses, mean, std




