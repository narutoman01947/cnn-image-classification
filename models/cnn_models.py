import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN with 2 convolutional layers.
    Baseline architecture for comparison.
    """
    def __init__(self, input_channels=1, num_classes=10, filters=None, kernel_size=3, dropout=0.25):
        super(SimpleCNN, self).__init__()
        if filters is None:
            filters = [32, 64]
        
        self.features = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(input_channels, filters[0], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv layer 2
            nn.Conv2d(filters[0], filters[1], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(filters[1] * 7 * 7, 128),  # Assumes 28x28 input -> 7x7 after 2 pooling
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MediumCNN(nn.Module):
    """
    Medium CNN with 3 convolutional layers.
    Includes batch normalization.
    """
    def __init__(self, input_channels=1, num_classes=10, filters=None, kernel_size=3, dropout=0.3, batch_norm=True):
        super(MediumCNN, self).__init__()
        if filters is None:
            filters = [32, 64, 128]
        
        self.batch_norm = batch_norm
        
        layers = []
        in_channels = input_channels
        
        # Build convolutional layers
        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate flattened size (28x28 -> 3x3 after 3 pooling operations)
        self.flat_size = filters[-1] * 3 * 3
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DeepCNN(nn.Module):
    """
    Deep CNN with 4 convolutional layers.
    Improved feature extraction with more layers.
    """
    def __init__(self, input_channels=1, num_classes=10, filters=None, kernel_size=3, dropout=0.4, batch_norm=True):
        super(DeepCNN, self).__init__()
        if filters is None:
            filters = [32, 64, 128, 256]
        
        self.batch_norm = batch_norm
        
        layers = []
        in_channels = input_channels
        
        # Build convolutional layers
        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate flattened size (28x28 -> 1x1 after 4 pooling operations)
        self.flat_size = filters[-1] * 1 * 1
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AdvancedCNN(nn.Module):
    """
    Advanced CNN with 5 convolutional layers.
    State-of-the-art architecture with advanced techniques.
    """
    def __init__(self, input_channels=1, num_classes=10, filters=None, kernel_size=3, dropout=0.5, batch_norm=True):
        super(AdvancedCNN, self).__init__()
        if filters is None:
            filters = [32, 64, 128, 256, 512]
        
        self.batch_norm = batch_norm
        
        layers = []
        in_channels = input_channels
        
        # Build convolutional layers
        for i, out_channels in enumerate(filters):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            # Add additional conv layer for first two stages for better feature extraction
            if i < 2:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(filters[-1], 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CustomCNN(nn.Module):
    """
    Customizable CNN that allows flexible architecture definition.
    """
    def __init__(self, input_channels=1, num_classes=10, conv_layers=3, filters=32, 
                 kernel_size=3, pool_size=2, dropout=0.3, batch_norm=True):
        super(CustomCNN, self).__init__()
        
        self.conv_layers = conv_layers
        self.batch_norm = batch_norm
        
        layers = []
        in_channels = input_channels
        current_filters = filters
        
        # Build convolutional layers
        for _ in range(conv_layers):
            layers.append(nn.Conv2d(in_channels, current_filters, kernel_size=kernel_size, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(current_filters))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
            
            in_channels = current_filters
            current_filters = min(current_filters * 2, 512)  # Double filters, max 512
        
        self.features = nn.Sequential(*layers)
        
        # Calculate flattened size
        # Assuming 28x28 input for MNIST
        self.flat_size = in_channels
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {
        'SimpleCNN': SimpleCNN(input_channels=1, num_classes=10),
        'MediumCNN': MediumCNN(input_channels=1, num_classes=10),
        'DeepCNN': DeepCNN(input_channels=1, num_classes=10),
        'AdvancedCNN': AdvancedCNN(input_channels=1, num_classes=10),
    }
    
    # Count parameters
    print("Model Complexity Analysis:")
    print("-" * 50)
    for name, model in models.items():
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Forward pass test
        x = torch.randn(1, 1, 28, 28).to(device)
        y = model(x)
        print(f"  Output shape: {y.shape}")
        print()
