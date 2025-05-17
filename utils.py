# utils.py
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# Import configurations
import config

def get_data_transforms():
    """Defines the transformations for training, validation, and test sets."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256), # Resize larger then crop
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def create_dataloaders(data_transforms):
    """Creates datasets and dataloaders for train, val, and test.
       Returns dataloaders and updates global config for NUM_CLASSES and CLASS_NAMES.
    """
    print(f"Looking for training data in: {config.TRAIN_DIR}")
    if not os.path.isdir(config.TRAIN_DIR):
        print(f"ERROR: Training directory not found: {config.TRAIN_DIR}")
        print("Please ensure your data is split and TRAIN_DIR is set correctly in config.py.")
        return None, None, None, None, None # Indicate error

    image_datasets = {}
    try:
        image_datasets['train'] = datasets.ImageFolder(config.TRAIN_DIR, data_transforms['train'])
        image_datasets['val'] = datasets.ImageFolder(config.VAL_DIR, data_transforms['val'])
        image_datasets['test'] = datasets.ImageFolder(config.TEST_DIR, data_transforms['test'])
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        print("Please check VAL_DIR and TEST_DIR in config.py.")
        return None, None, None, None, None # Indicate error


    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=config.BATCH_SIZE,
                      shuffle=(x == 'train'), num_workers=config.NUM_WORKERS)
        for x in ['train', 'val', 'test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    # Dynamically set NUM_CLASSES and CLASS_NAMES from the training dataset
    # This is important if these are not hardcoded in config.py
    class_names_from_dataset = image_datasets['train'].classes
    num_classes_from_dataset = len(class_names_from_dataset)

    print(f"Found {num_classes_from_dataset} classes: {class_names_from_dataset}")
    print(f"Dataset sizes: Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}, Test: {dataset_sizes['test']}")

    return (dataloaders['train'], dataloaders['val'], dataloaders['test'],
            class_names_from_dataset, num_classes_from_dataset)


def get_model(num_classes: int):
    """Defines or loads a pre-trained model."""
    # MODIFY: You can change this to use your custom model or a different pretrained one.
    # Example using a pretrained ResNet18
    if config.PRETRAINED_MODEL:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET18_V1)
    else:
        model = models.resnet18(weights=None) # or weights=None for PyTorch 1.9+

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes) # Replace the final layer
    return model.to(config.DEVICE)