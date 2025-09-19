import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import time
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to select GPU with maximum free memory
def get_best_gpu():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    try:
        # Clear CUDA cache first
        torch.cuda.empty_cache()
        
        # Get free memory for each GPU
        free_memory = []
        for i in range(torch.cuda.device_count()):
            try:
                # Get total and allocated memory
                total_mem = torch.cuda.get_device_properties(i).total_memory
                allocated_mem = torch.cuda.memory_allocated(i)
                free_mem = total_mem - allocated_mem
                free_memory.append((i, free_mem))
            except RuntimeError:
                print(f"Warning: Could not get memory info for GPU {i}")
                continue
        
        if not free_memory:
            print("Warning: No GPUs available with memory info, defaulting to CPU")
            return torch.device('cpu')
        
        # Select GPU with maximum free memory
        best_gpu_id = max(free_memory, key=lambda x: x[1])[0]
        print(f"Selected GPU {best_gpu_id} with {free_memory[best_gpu_id][1] / 1024**3:.2f} GB free memory")
        return torch.device(f'cuda:{best_gpu_id}')
        
    except Exception as e:
        print(f"Warning: Error selecting GPU: {str(e)}")
        print("Defaulting to CPU")
        return torch.device('cpu')

# Define paths
train_dir = "/home/shashwat/LoRa/data/train"
val_dir = "/home/shashwat/LoRa/data/val"
test_dir = "/home/shashwat/LoRa/data/test"
results_dir = "results"

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

# Define model architecture
class LoRaClassifier(nn.Module):
    def __init__(self):
        super(LoRaClassifier, self).__init__()
        # Convolutional Block 1: 3→16→32 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Convolutional Block 2: 32→64→128 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Convolutional Block 3: 128→256→256 channels
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Pooling and classifier layers
        self.pool = nn.MaxPool2d(2, 2)               # 2x2 max pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))     # global average pooling to 1x1
        self.dropout = nn.Dropout(0.5)                  # dropout for regularization
        self.fc = nn.Linear(256, 2)                # fully-connected output (2 classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)        # flatten to (batch_size, 256)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Data transforms (without adding noise since it's already in the data)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
}

# Function to save the model
def save_model(model, path):
    """Save model weights to file"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load model
def load_model(model, path, device):
    """Load model weights from file"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")
    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               num_epochs, device, model_save_path):
    """Train the model and validate after each epoch"""
    since = time.time()
    best_acc = 0.0
    best_epoch = 0
    
    # Lists to store metrics history for plotting
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    # Learning rate history
    lr_history = []
    
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc)
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Record learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Step the scheduler
        scheduler.step()
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            save_model(model, model_save_path)
            
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch}')
    
    # Return history for plotting
    history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'lr': lr_history,
        'best_epoch': best_epoch,
        'best_acc': best_acc
    }
    return history

# Evaluation function for single data loader
def evaluate_model(model, data_loader, criterion, device):
    """Evaluate the model on a data loader"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """Plot confusion matrix and optionally save to file"""
    # Check if arrays are empty
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: Empty arrays provided for confusion matrix")
        return None
        
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Check if confusion matrix is empty
    if cm.size == 0:
        print("Warning: Empty confusion matrix")
        return None
        
    # Create plot
    plt.figure(figsize=(8, 6))
    try:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
    except Exception as e:
        print(f"Warning: Error plotting confusion matrix: {str(e)}")
    finally:
        plt.close()
    
    return cm

# Function to plot learning curves
def plot_learning_curves(history, save_dir):
    """Plot training and validation loss and accuracy curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-', label='Training Accuracy')
    plt.plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    print(f"Learning curves saved to {os.path.join(save_dir, 'learning_curves.png')}")
    plt.close()
    
    # Plot learning rate curve
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['lr'], 'g-')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'))
    print(f"Learning rate curve saved to {os.path.join(save_dir, 'learning_rate.png')}")
    plt.close()

# Function to evaluate model on test data with different SNR levels
def evaluate_on_snr_levels(model, criterion, device, test_dir, transform, snr_levels=None):
    """Evaluate model on different SNR levels"""
    if snr_levels is None:
        # Default SNR levels in the dataset
        snr_levels = ["-30dB", "-20dB", "-10dB", "0dB", "10dB"]
    
    results = {}
    all_preds = []
    all_labels = []
    
    for snr in snr_levels:
        snr_dir = os.path.join(test_dir, snr)
        if not os.path.isdir(snr_dir):
            print(f"Warning: Test directory for SNR {snr} not found: {snr_dir}")
            continue
        
        # Load data for this SNR level
        test_dataset = datasets.ImageFolder(snr_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Evaluate model
        loss, acc, preds, labels = evaluate_model(model, test_loader, criterion, device)
        results[snr] = {'loss': loss, 'acc': acc, 'preds': preds, 'labels': labels}
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        print(f"SNR {snr} - Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%")
    
    return results, all_preds, all_labels

# Function to plot accuracy vs SNR
def plot_accuracy_vs_snr(results, save_path):
    """Plot accuracy vs SNR and save to file"""
    if not results:
        print("Warning: No results available for plotting accuracy vs SNR")
        return None, None
        
    snr_levels = sorted(list(results.keys()), key=lambda x: int(x.replace('dB', '')))
    accuracies = [results[snr]['acc'] * 100 for snr in snr_levels]
    
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.title('LoRa Classification Accuracy vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.xticks(snr_levels)
    plt.ylim(0, 100)
    
    # Add data values on points
    for i, acc in enumerate(accuracies):
        plt.annotate(f'{acc:.1f}%', 
                    (snr_levels[i], accuracies[i]), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Accuracy vs SNR plot saved to {save_path}")
    plt.close()
    
    return snr_levels, accuracies

# Export model to ONNX
def export_to_onnx(model, save_path, input_shape=(1, 3, 300, 300)):
    """Export model to ONNX format"""
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape, device=device)
    
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            save_path, 
            input_names=["input"], 
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11
        )
        print(f"Model exported to ONNX format at: {save_path}")
    except Exception as e:
        print(f"Warning: Failed to export model to ONNX: {str(e)}")

# Main function
def main():
    # Select device (GPU with maximum free memory)
    device = get_best_gpu()
    print(f"Using device: {device}")
    
    # Create timestamp for model saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(results_dir, f"lora_classifier_{timestamp}.pth")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = LoRaClassifier().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Train the model
    print("\nTraining model...")
    num_epochs = 100
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs, device, model_save_path
    )
    
    # Plot learning curves
    plot_learning_curves(history, results_dir)
    
    # Load best model for evaluation
    best_model = LoRaClassifier()
    best_model = load_model(best_model, model_save_path, device)
    
    # Evaluate on different SNR levels
    print("\nEvaluating on test data across SNR levels...")
    snr_results, all_preds, all_labels = evaluate_on_snr_levels(
        best_model, criterion, device, test_dir, data_transforms['test']
    )
    
    # Plot accuracy vs SNR
    snr_levels, accuracies = plot_accuracy_vs_snr(
        snr_results, os.path.join(results_dir, 'accuracy_vs_SNR.png')
    )
    
    # Plot confusion matrix for all test data
    if len(all_preds) > 0 and len(all_labels) > 0:
        plot_confusion_matrix(
            all_labels, all_preds, class_names,
            save_path=os.path.join(results_dir, 'confusion_matrix.png')
        )
    
    # Generate and save classification report only if we have predictions
    if len(all_preds) > 0 and len(all_labels) > 0:
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        print("\nClassification Report:")
        print(report)
    else:
        print("\nWarning: No predictions available for classification report")
    
    # Export model to ONNX
    try:
        export_to_onnx(best_model, os.path.join(results_dir, 'lora_classifier.onnx'))
    except Exception as e:
        print(f"Warning: Failed to export model to ONNX: {str(e)}")
    
    # Save additional metrics to file
    metrics = {
        'snr_levels': snr_levels if snr_levels else [],
        'accuracies': accuracies if accuracies else [],
        'best_val_accuracy': history['best_acc'],
        'best_epoch': history['best_epoch'],
        'total_parameters': total_params
    }
    
    # Save metrics to text file
    with open(os.path.join(results_dir, 'model_metrics.txt'), 'w') as f:
        f.write(f"Model: LoRa vs Non-LoRa Classifier\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Best Validation Accuracy: {history['best_acc']*100:.2f}% (Epoch {history['best_epoch']})\n\n")
        f.write(f"SNR Level Accuracies:\n")
        if snr_levels and accuracies:
            for snr, acc in zip(snr_levels, accuracies):
                f.write(f"  SNR {snr}: {acc:.2f}%\n")
        else:
            f.write("  No SNR level results available\n")
    
    print("\nTraining and evaluation complete. All results saved to the 'results' directory.")

if __name__ == "__main__":
    main()