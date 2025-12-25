import numpy as np
import torch
import wandb
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from fashion_mnist_dataset import load_fashion_data
from fashion_mnist_model import MultiTaskFashionCNN , calculate_joint_loss


# --- 2. METRICS FUNCTIONS ---

def calculate_metrics(logits, predictions, labels, targets):
    """Calculates classification accuracy and regression MAE/RMSE."""
    # Classification Metrics (Accuracy)
    _, predicted_classes = torch.max(logits, 1)
    correct_predictions = (predicted_classes == labels).sum().item()
    accuracy = correct_predictions / labels.size(0)
    
    # Regression Metrics (MAE, RMSE)
    # Ensure targets and predictions are flat for metric calculation
    targets = targets.view(-1).cpu().numpy()
    predictions = predictions.view(-1).cpu().numpy()
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets)**2))
    
    return accuracy, mae, rmse

# --- 4. TEST EVALUATION FUNCTION ---

def evaluate_test_set(model, test_loader, device, lambda1, lambda2):
    """Evaluates the final model state on the test set."""
    model.eval()
    test_loss_sum, test_ce_loss_sum, test_mse_loss_sum = 0, 0, 0
    test_total_correct, test_mae_sum, test_rmse_sum = 0, 0, 0
    test_total_samples = 0
    
    with torch.no_grad():
        for images, labels, ink_targets in test_loader:
            images, labels, ink_targets = images.to(device), labels.to(device), ink_targets.to(device)
            
            logits, predictions = model(images)
            
            total_loss, L_CE, L_MSE = calculate_joint_loss(
                logits, labels, predictions, ink_targets, lambda1, lambda2
            )
            
            test_loss_sum += total_loss.item() * images.size(0)
            
            # Calculate Metrics
            accuracy, mae, rmse = calculate_metrics(logits, predictions, labels, ink_targets)
            test_total_correct += accuracy * images.size(0)
            test_mae_sum += mae * images.size(0)
            test_rmse_sum += rmse * images.size(0) # Simple batch RMSE sum (approximation)
            test_total_samples += images.size(0)

    # Calculate average test metrics
    avg_test_accuracy = test_total_correct / test_total_samples
    avg_test_mae = test_mae_sum / test_total_samples
    avg_test_rmse = test_rmse_sum / test_total_samples 
    
    return {
        "accuracy": avg_test_accuracy,
        "mae": avg_test_mae,
        "rmse": avg_test_rmse,
        "loss_total": test_loss_sum / test_total_samples
    }

# --- 3. TRAINING AND VALIDATION LOOP ---

def train_and_validate(config=None):
    """
    Runs a single training experiment with specified hyperparameters and logs to wandb.
    """
    # 1. Initialize wandb run
    # 'reinit=True' allows running multiple experiments in one notebook session
    wandb.init(config=config, project="MultiTask-FashionMNIST-CNN", reinit=True)
    config = wandb.config # Access the configuration object
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, val_loader, test_loader = load_fashion_data(
        batch_size=config.batch_size
    )

    # 2. Initialize Model, Optimizer, and Scheduler
    model = MultiTaskFashionCNN(
        num_classes=10, 
        dropout_rate=config.dropout_rate
    ).to(device)

    # Optimizer selection
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    
    # Best validation metric trackers for model selection
    best_val_accuracy = 0.0
    best_val_rmse = float('inf')
    
    # --- Training Loop ---
    for epoch in range(config.epochs):
        model.train()
        train_loss_sum, train_ce_loss_sum, train_mse_loss_sum = 0, 0, 0
        
        # 3. Training Step
        for images, labels, ink_targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images, labels, ink_targets = images.to(device), labels.to(device), ink_targets.to(device)
            
            optimizer.zero_grad()
            logits, predictions = model(images)
            
            # Calculate Joint Loss (using config lambda values)
            total_loss, L_CE, L_MSE = calculate_joint_loss(
                logits, labels, predictions, ink_targets, 
                lambda1=config.lambda1, lambda2=config.lambda2
            )
            
            total_loss.backward()
            optimizer.step()
            
            train_loss_sum += total_loss.item() * images.size(0)
            train_ce_loss_sum += L_CE.item() * images.size(0)
            train_mse_loss_sum += L_MSE.item() * images.size(0)

        # Calculate average training losses
        avg_train_loss = train_loss_sum / len(train_loader.dataset)
        avg_train_ce_loss = train_ce_loss_sum / len(train_loader.dataset)
        avg_train_mse_loss = train_mse_loss_sum / len(train_loader.dataset)
        
        # 4. Validation Step
        model.eval()
        val_loss_sum, val_ce_loss_sum, val_mse_loss_sum = 0, 0, 0
        val_total_correct, val_mae_sum, val_rmse_sum = 0, 0, 0
        val_total_samples = 0
        
        with torch.no_grad():
            for images, labels, ink_targets in val_loader:
                images, labels, ink_targets = images.to(device), labels.to(device), ink_targets.to(device)
                
                logits, predictions = model(images)
                
                total_loss, L_CE, L_MSE = calculate_joint_loss(
                    logits, labels, predictions, ink_targets, 
                    lambda1=config.lambda1, lambda2=config.lambda2
                )
                
                val_loss_sum += total_loss.item() * images.size(0)
                val_ce_loss_sum += L_CE.item() * images.size(0)
                val_mse_loss_sum += L_MSE.item() * images.size(0)
                
                # Calculate Metrics
                accuracy, mae, rmse = calculate_metrics(logits, predictions, labels, ink_targets)
                val_total_correct += accuracy * images.size(0)
                val_mae_sum += mae * images.size(0)
                val_rmse_sum += rmse * images.size(0) # Note: this is an approximation; true RMSE needs summing squared errors.
                                                     # For simplicity here, we approximate by summing batch RMSE.
                val_total_samples += images.size(0)
        
        # Calculate average validation metrics
        avg_val_loss = val_loss_sum / val_total_samples
        avg_val_ce_loss = val_ce_loss_sum / val_total_samples
        avg_val_mse_loss = val_mse_loss_sum / val_total_samples
        avg_val_accuracy = val_total_correct / val_total_samples
        avg_val_mae = val_mae_sum / val_total_samples
        avg_val_rmse = avg_val_mse_loss**0.5 # A more accurate way to use the average MSE for RMSE (still a slight approx)
                                            # The simpler `val_rmse_sum / val_total_samples` is also acceptable.

        # 5. wandb Logging
        wandb.log({
            "epoch": epoch,
            # Loss curves
            "train/loss_total": avg_train_loss,
            "train/loss_ce": avg_train_ce_loss,
            "train/loss_mse": avg_train_mse_loss,
            "val/loss_total": avg_val_loss,
            "val/loss_ce": avg_val_ce_loss,
            "val/loss_mse": avg_val_mse_loss,
            # Validation metrics
            "val/accuracy": avg_val_accuracy,
            "val/mae": avg_val_mae,
            "val/rmse": avg_val_rmse,
        })
        
        # 6. Model Selection (Save checkpoints based on best performance)
        # Checkpoint for best accuracy
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), f"best_accuracy_model_lambda{config.lambda1}_{config.lambda2}.pth")
            wandb.run.summary["best_val_accuracy"] = best_val_accuracy
            wandb.run.summary["epoch_best_accuracy"] = epoch

        # Checkpoint for best RMSE
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            torch.save(model.state_dict(), f"best_rmse_model_lambda{config.lambda1}_{config.lambda2}.pth")
            wandb.run.summary["best_val_rmse"] = best_val_rmse
            wandb.run.summary["epoch_best_rmse"] = epoch

    # 7. Final Test Evaluation (for the last model state)
    final_test_metrics = evaluate_test_set(model, test_loader, device, config.lambda1, config.lambda2)
    wandb.log({"test": final_test_metrics})
    
    wandb.finish()
    
    return best_val_accuracy, best_val_rmse