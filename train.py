import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
from datetime import datetime
import argparse

from data.data_loader import DataManager
from models import SimpleCNN, MediumCNN, DeepCNN, AdvancedCNN, CustomCNN


class Trainer:
    """Trainer class for CNN models."""
    
    def __init__(self, model, device, learning_rate=0.001, weight_decay=0.0005, 
                 checkpoint_dir="./results/models"):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            device: torch device (cuda or cpu)
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=0.00001)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': None,
            'test_acc': None
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss / total, 'acc': correct / total})
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({'loss': total_loss / total, 'acc': correct / total})
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
    
    def test(self, test_loader):
        """Test model on test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                
                pbar.set_postfix({'loss': total_loss / total, 'acc': correct / total})
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=50, patience=10):
        """
        Train model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            patience: Early stopping patience
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"\n{'Epoch':<8} {'Train Loss':<15} {'Train Acc':<15} {'Val Loss':<15} {'Val Acc':<15}")
        print("-" * 65)
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"{epoch+1:<8} {train_loss:<15.4f} {train_acc:<15.4f} {val_loss:<15.4f} {val_acc:<15.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print("-" * 65)
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded: {filepath}")
    
    def save_history(self, filename):
        """Save training history as JSON."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"History saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN models")
    parser.add_argument("--model", type=str, default="SimpleCNN", 
                       choices=["SimpleCNN", "MediumCNN", "DeepCNN", "AdvancedCNN", "CustomCNN"],
                       help="Model architecture to train")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR-10"],
                       help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay (L2)")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    # Custom model parameters
    parser.add_argument("--conv_layers", type=int, default=3, help="Number of convolutional layers (CustomCNN)")
    parser.add_argument("--filters", type=int, default=32, help="Initial number of filters (CustomCNN)")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size (CustomCNN)")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    data_manager = DataManager(args.dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    dataloaders = data_manager.get_dataloaders(augment=args.augment)
    
    # Create model
    print(f"Creating {args.model} model...")
    num_input_channels = 1 if args.dataset == "MNIST" else 3
    
    if args.model == "SimpleCNN":
        model = SimpleCNN(input_channels=num_input_channels, num_classes=10)
    elif args.model == "MediumCNN":
        model = MediumCNN(input_channels=num_input_channels, num_classes=10)
    elif args.model == "DeepCNN":
        model = DeepCNN(input_channels=num_input_channels, num_classes=10)
    elif args.model == "AdvancedCNN":
        model = AdvancedCNN(input_channels=num_input_channels, num_classes=10)
    elif args.model == "CustomCNN":
        model = CustomCNN(input_channels=num_input_channels, num_classes=10,
                         conv_layers=args.conv_layers, filters=args.filters,
                         kernel_size=args.kernel_size)
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {trainable_params:,} / {total_params:,}")
    
    # Create trainer and train
    trainer = Trainer(model, device, learning_rate=args.learning_rate, 
                     weight_decay=args.weight_decay)
    
    print(f"\nTraining started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    trainer.train(dataloaders['train'], dataloaders['val'], 
                 num_epochs=args.epochs, patience=args.patience)
    
    # Test
    print(f"\nTesting on {args.dataset} test set...")
    test_loss, test_acc, predictions, labels = trainer.test(dataloaders['test'])
    trainer.history['test_loss'] = test_loss
    trainer.history['test_acc'] = test_acc
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Save results
    model_name = f"{args.model}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer.save_checkpoint(f"{model_name}_final.pt")
    trainer.save_history(f"{model_name}_history.json")
    
    print(f"\nTraining completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model name: {model_name}")


if __name__ == "__main__":
    main()
