import os
import torch
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from glob import glob

from data.data_loader import DataManager
from models import SimpleCNN, MediumCNN, DeepCNN, AdvancedCNN


class Evaluator:
    """Evaluator class for analyzing model performance."""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu",
                 plots_dir="./results/plots"):
        self.device = device
        self.plots_dir = plots_dir
        os.makedirs(plots_dir, exist_ok=True)
    
    def compute_metrics(self, predictions, labels, class_names=None):
        """
        Compute comprehensive metrics.
        
        Args:
            predictions: List of predicted labels
            labels: List of true labels
            class_names: List of class names
            
        Returns:
            dict: Dictionary with computed metrics
        """
        accuracy = accuracy_score(labels, predictions)
        conf_matrix = confusion_matrix(labels, predictions)
        class_report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        return metrics
    
    def plot_confusion_matrix(self, predictions, labels, dataset_name, model_name, class_names=None):
        """
        Plot and save confusion matrix.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            dataset_name: Name of dataset
            model_name: Name of model
            class_names: List of class names
        """
        conf_matrix = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        filename = f"{model_name}_{dataset_name}_confusion_matrix.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_training_curves(self, history, model_name, dataset_name):
        """
        Plot training and validation curves.
        
        Args:
            history: Dictionary with training history
            model_name: Name of model
            dataset_name: Name of dataset
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Loss Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{model_name} - Accuracy Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"{model_name}_{dataset_name}_training_curves.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_model_comparison(self, results_dict, metric='accuracy', dataset_name='MNIST'):
        """
        Plot comparison of multiple models.
        
        Args:
            results_dict: Dictionary with {model_name: metric_value}
            metric: Metric to compare
            dataset_name: Name of dataset
        """
        models = list(results_dict.keys())
        values = list(results_dict.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.ylabel(metric.capitalize())
        plt.title(f'Model Comparison - {metric.capitalize()} on {dataset_name}')
        plt.ylim([min(values) * 0.95, max(values) * 1.05])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filename = f"{dataset_name}_model_comparison_{metric}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def generate_report(self, results, output_file=None):
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Dictionary with evaluation results
            output_file: Path to save report
        """
        report_text = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    CNN IMAGE CLASSIFICATION REPORT                         ║
╚══════════════════════════════════════════════════════════════════��═════════╝

Report Generated: {}

""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        for model_name, model_results in results.items():
            report_text += f"\n{'='*80}\n"
            report_text += f"Model: {model_name}\n"
            report_text += f"Dataset: {model_results.get('dataset', 'N/A')}\n"
            report_text += f"{'='*80}\n"
            
            if 'test_accuracy' in model_results:
                report_text += f"Test Accuracy: {model_results['test_accuracy']:.4f} ({model_results['test_accuracy']*100:.2f}%)\n"
            if 'test_loss' in model_results:
                report_text += f"Test Loss: {model_results['test_loss']:.4f}\n"
            if 'num_parameters' in model_results:
                report_text += f"Number of Parameters: {model_results['num_parameters']:,}\n"
            if 'training_time' in model_results:
                report_text += f"Training Time: {model_results['training_time']:.2f}s\n"
            
            report_text += f"\nPer-Class Performance:\n"
            report_text += f"{'-'*80}\n"
            
            if 'classification_report' in model_results:
                for class_idx, metrics in model_results['classification_report'].items():
                    if class_idx not in ['accuracy', 'macro avg', 'weighted avg']:
                        report_text += f"Class {class_idx}: "
                        report_text += f"Precision={metrics['precision']:.4f}, "
                        report_text += f"Recall={metrics['recall']:.4f}, "
                        report_text += f"F1={metrics['f1-score']:.4f}\n"
        
        print(report_text)
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_file}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN models")
    parser.add_argument("--models_dir", type=str, default="./results/models",
                       help="Directory containing saved models")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR-10"],
                       help="Dataset to evaluate on")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--compare_all", action="store_true",
                       help="Compare all saved models")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    evaluator = Evaluator(device=device)
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    data_manager = DataManager(args.dataset, batch_size=args.batch_size, 
                              num_workers=args.num_workers)
    dataloaders = data_manager.get_dataloaders()
    class_names = data_manager.get_class_names()
    
    print(f"Dataset info:")
    print(f"  Classes: {len(class_names)}")
    print(f"  Test samples: {len(dataloaders['test'].dataset)}")
    
    # Collect results
    all_results = {}
    
    # Model definitions
    num_input_channels = 1 if args.dataset == "MNIST" else 3
    models_def = {
        'SimpleCNN': SimpleCNN(input_channels=num_input_channels, num_classes=10),
        'MediumCNN': MediumCNN(input_channels=num_input_channels, num_classes=10),
        'DeepCNN': DeepCNN(input_channels=num_input_channels, num_classes=10),
        'AdvancedCNN': AdvancedCNN(input_channels=num_input_channels, num_classes=10),
    }
    
    print(f"\n{'Model':<20} {'Parameters':<15} {'Test Accuracy':<15}")
    print("-" * 50)
    
    for model_name, model_template in models_def.items():
        model_template.to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model_template.parameters())
        
        # For demonstration, create a mock result
        # In practice, you would load actual trained model checkpoints
        result = {
            'dataset': args.dataset,
            'num_parameters': num_params,
        }
        
        all_results[model_name] = result
        print(f"{model_name:<20} {num_params:<15,} {'See logs':<15}")
    
    print("-" * 50)
    
    # Generate report
    report_path = os.path.join("./results/reports", 
                              f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    evaluator.generate_report(all_results, output_file=report_path)


if __name__ == "__main__":
    main()
