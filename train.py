# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import copy
from tqdm import tqdm
from multiprocessing import freeze_support

# Import configurations and utilities
import config
import utils

def train_model_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    """Trains the model."""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\n--- Epoch {epoch+1}/{num_epochs} ---')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            num_samples_phase = 0

            for inputs, labels in tqdm(dataloader, desc=f"{phase.capitalize()}"):
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_samples_phase += inputs.size(0)

            epoch_loss = running_loss / num_samples_phase
            epoch_acc = running_corrects.double() / num_samples_phase

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item()) # Store as float

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_save_path)
                print(f"Validation accuracy improved. Model saved to {model_save_path}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts) # load best model weights
    return model, history

# --- Main Execution Block for Training ---
if __name__ == '__main__':
    freeze_support() # Crucial for multiprocessing (DataLoader num_workers > 0)

    print(f"Using device: {config.DEVICE}")
    print(f"Number of workers for DataLoader: {config.NUM_WORKERS}")

    # Create output directory if it doesn't exist
    if not os.path.exists(config.BASE_OUTPUT_DIR):
        os.makedirs(config.BASE_OUTPUT_DIR)
        print(f"Created output directory: {config.BASE_OUTPUT_DIR}")

    # 1. Get Data Transforms
    data_transforms = utils.get_data_transforms()

    # 2. Create DataLoaders
    # This will also set NUM_CLASSES and CLASS_NAMES in the config module if not hardcoded
    # For train.py, we primarily need train_loader and val_loader
    train_loader, val_loader, _, class_names_from_data, num_classes_from_data = utils.create_dataloaders(
        data_transforms
    )

    if train_loader is None:
        print("Exiting due to dataloader creation errors. Check config.py and data paths.")
        exit()

    # Update config with dynamically determined class info if they weren't set
    if config.NUM_CLASSES is None:
        config.NUM_CLASSES = num_classes_from_data
    if config.CLASS_NAMES is None:
        config.CLASS_NAMES = class_names_from_data


    # 3. Define the Model
    if config.NUM_CLASSES is None: # Should be set by create_dataloaders
        print("ERROR: NUM_CLASSES was not set. Cannot define model. Check create_dataloaders in utils.py")
        exit()
    model_pytorch = utils.get_model(num_classes=config.NUM_CLASSES) # Use the updated NUM_CLASSES
    print(f"\nModel: {type(model_pytorch).__name__} loaded with {config.NUM_CLASSES} output classes.")

    # 4. Define Criterion (Loss Function) and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_pytorch.parameters(), lr=config.LEARNING_RATE)

    # 5. Train the Model
    print("\nStarting model training...")
    model_pytorch, training_history = train_model_loop(
        model_pytorch, train_loader, val_loader,
        criterion, optimizer, config.NUM_EPOCHS, config.MODEL_SAVE_PATH
    )
    print("\n--- Training Script Finished ---")
    # Optionally save or plot training_history here