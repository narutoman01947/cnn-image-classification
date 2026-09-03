# CNN Image Classification Project

A comprehensive implementation of Convolutional Neural Networks (CNNs) for image classification with experiments on MNIST and CIFAR-10 datasets.

## Project Overview

This project explores CNN architectures and their impact on model performance through:
- **Multiple CNN architectures** with varying depths and configurations
- **MNIST dataset** (handwritten digits) classification
- **CIFAR-10 dataset** (colored objects) classification
- **Detailed performance analysis** and architectural comparisons
- **Visualization tools** for training metrics and predictions

## Features

### Implemented Architectures

1. **SimpleCNN**: Basic 2-layer convolutional network
   - 2 convolutional layers
   - Simple pooling strategy
   - Baseline performance reference

2. **MediumCNN**: Intermediate depth network
   - 3 convolutional layers
   - Progressive filter increase
   - Batch normalization

3. **DeepCNN**: Deep convolutional network
   - 4-5 convolutional layers
   - Large filter bank
   - Dropout regularization

4. **AdvancedCNN**: Advanced architecture with modern techniques
   - ResNet-inspired skip connections (optional)
   - Adaptive pooling
   - Advanced regularization

### Experimental Parameters

- **Filter sizes**: 3×3, 5×5, 7×7
- **Pooling operations**: Max pooling, Average pooling
- **Depths**: 2, 3, 4, 5 convolutional layers
- **Batch sizes**: 32, 64, 128
- **Learning rates**: 0.001, 0.0005, 0.0001

## Dataset Information

### MNIST
- 60,000 training images
- 10,000 test images
- 28×28 grayscale digits
- 10 classes (0-9)

### CIFAR-10
- 50,000 training images
- 10,000 test images
- 32×32 RGB images
- 10 classes (airplane, automobile, bird, cat, etc.)

## Installation

```bash
# Clone the repository
git clone https://github.com/narutoman01947/cnn-image-classification.git
cd cnn-image-classification

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train a Model

```bash
# Train SimpleCNN on MNIST
python train.py --model SimpleCNN --dataset MNIST --epochs 20 --batch_size 64

# Train DeepCNN on CIFAR-10
python train.py --model DeepCNN --dataset CIFAR-10 --epochs 50 --batch_size 32

# Custom architecture
python train.py --model CustomCNN --dataset MNIST --epochs 30 --conv_layers 3 --filters 32 --kernel_size 5
```

### Evaluate Models

```bash
# Evaluate all trained models
python evaluate.py --compare_all

# Detailed analysis of specific model
python evaluate.py --model DeepCNN --dataset CIFAR-10 --detailed
```

### Visualize Results

```bash
# Plot training curves
python visualize.py --model SimpleCNN --metric loss

# Confusion matrix
python visualize.py --model DeepCNN --confusion_matrix

# Prediction samples
python visualize.py --model MediumCNN --predictions --count 16
```

## Project Structure

```
cnn-image-classification/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── download_datasets.py
│   └── data_loader.py
├── models/
│   ├── __init__.py
│   ├── simple_cnn.py
│   ├── medium_cnn.py
│   ├── deep_cnn.py
│   └── advanced_cnn.py
├── train.py
├── evaluate.py
├── visualize.py
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── helpers.py
├── results/
│   ├── models/
│   ├── plots/
│   └── reports/
└── experiments/
    ├── baseline_experiments.py
    ├── architecture_comparison.py
    └── hyperparameter_tuning.py
```

## Experimental Results

See `results/reports/` for detailed analysis including:
- **Accuracy Comparison**: Performance across architectures
- **Training Curves**: Loss and accuracy over epochs
- **Computational Cost**: Training time and memory usage
- **Effect of Architecture**: Impact of depth, filters, kernel size
- **Dataset Comparison**: MNIST vs CIFAR-10 performance

## Key Findings

### Effect of CNN Depth
- Deeper networks generally improve performance
- Diminishing returns after 4-5 layers
- Risk of overfitting with very deep networks on MNIST

### Effect of Filter Size
- 3×3 filters: Efficient, requires more layers
- 5×5 filters: Balance between receptive field and efficiency
- 7×7 filters: Larger receptive field but increased parameters

### Effect of Pooling
- Max pooling: Better for feature preservation
- Average pooling: Smoother feature maps
- Hybrid approaches show mixed results

### Dataset Difficulty
- MNIST: Simple task, convergence in 5-10 epochs
- CIFAR-10: Complex task, requires 30-50 epochs for good performance

## Performance Benchmarks

| Model | Dataset | Accuracy | Training Time |
|-------|---------|----------|----------------|
| SimpleCNN | MNIST | 97.2% | 2 min |
| MediumCNN | MNIST | 98.5% | 4 min |
| DeepCNN | MNIST | 99.1% | 7 min |
| SimpleCNN | CIFAR-10 | 65.3% | 5 min |
| MediumCNN | CIFAR-10 | 72.1% | 9 min |
| DeepCNN | CIFAR-10 | 76.8% | 15 min |

## Best Practices Demonstrated

✓ Proper train/validation/test splits
✓ Data normalization and augmentation
✓ Batch normalization for faster training
✓ Dropout for regularization
✓ Proper learning rate scheduling
✓ Model checkpointing and persistence
✓ Comprehensive evaluation metrics
✓ Visualization of results

## Improvements and Future Work

1. **Data Augmentation**: Rotation, zoom, horizontal flip for CIFAR-10
2. **Regularization**: L1/L2 penalties, early stopping
3. **Transfer Learning**: Pre-trained models for CIFAR-10
4. **Architecture Search**: AutoML for optimal architecture
5. **Ensemble Methods**: Combine multiple models
6. **Mobile Optimization**: Model compression and quantization

## References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning (MIT Press)

## License

MIT License - See LICENSE file for details

## Author

narutoman01947

## Contributions

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
