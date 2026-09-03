import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class DataManager:
    """Manages data loading for MNIST and CIFAR-10 datasets."""
    
    def __init__(self, dataset_name="MNIST", batch_size=64, data_dir="./data", num_workers=4):
        """
        Initialize DataManager.
        
        Args:
            dataset_name (str): "MNIST" or "CIFAR-10"
            batch_size (int): Batch size for training
            data_dir (str): Directory to store datasets
            num_workers (int): Number of workers for data loading
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        
        os.makedirs(data_dir, exist_ok=True)
        
        if dataset_name == "MNIST":
            self.num_classes = 10
            self.input_shape = (1, 28, 28)
            self.mean = (0.1307,)
            self.std = (0.3081,)
        elif dataset_name == "CIFAR-10":
            self.num_classes = 10
            self.input_shape = (3, 32, 32)
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def get_transforms(self, augment=False):
        """
        Get data transforms.
        
        Args:
            augment (bool): Whether to apply data augmentation
            
        Returns:
            tuple: (train_transforms, test_transforms)
        """
        if self.dataset_name == "MNIST":
            if augment:
                train_transform = transforms.Compose([
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        
        elif self.dataset_name == "CIFAR-10":
            if augment:
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
            
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        
        return train_transform, test_transform
    
    def get_dataloaders(self, augment=False, validation_split=0.2):
        """
        Get train, validation, and test dataloaders.
        
        Args:
            augment (bool): Whether to apply data augmentation
            validation_split (float): Fraction of training data to use for validation
            
        Returns:
            dict: Dictionary with 'train', 'val', and 'test' dataloaders
        """
        train_transform, test_transform = self.get_transforms(augment=augment)
        
        if self.dataset_name == "MNIST":
            train_dataset = torchvision.datasets.MNIST(
                root=self.data_dir,
                train=True,
                transform=train_transform,
                download=True
            )
            test_dataset = torchvision.datasets.MNIST(
                root=self.data_dir,
                train=False,
                transform=test_transform,
                download=True
            )
        
        elif self.dataset_name == "CIFAR-10":
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=train_transform,
                download=True
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=test_transform,
                download=True
            )
        
        # Split training data into train and validation
        train_size = int(len(train_dataset) * (1 - validation_split))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def get_class_names(self):
        """Get class names for the dataset."""
        if self.dataset_name == "MNIST":
            return [str(i) for i in range(10)]
        elif self.dataset_name == "CIFAR-10":
            return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


if __name__ == "__main__":
    # Test data loading
    print("Testing MNIST data loading...")
    mnist_manager = DataManager("MNIST", batch_size=64)
    loaders = mnist_manager.get_dataloaders()
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    # Get a batch
    images, labels = next(iter(loaders['train']))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
    
    print("\nTesting CIFAR-10 data loading...")
    cifar_manager = DataManager("CIFAR-10", batch_size=64)
    loaders = cifar_manager.get_dataloaders()
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    images, labels = next(iter(loaders['train']))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
