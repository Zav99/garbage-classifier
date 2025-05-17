# evaluate.py
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from multiprocessing import freeze_support

# Import configurations and utilities
import config
import utils

def evaluate_model_on_test_set(model, test_loader, criterion, class_names_list):
    """Evaluates the model on the test set and prints metrics."""
    print("\nStarting evaluation on the Test Set...")
    model.eval()
    model.to(config.DEVICE) # Ensure model is on the correct device

    test_loss = 0.0
    test_corrects = 0
    test_samples_count = 0

    all_preds_np = []
    all_labels_np = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            test_samples_count += labels.size(0)

            all_preds_np.extend(preds.cpu().numpy())
            all_labels_np.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / test_samples_count
    avg_test_acc = test_corrects.double() / test_samples_count

    print(f"\n--- Test Set Evaluation Complete ---")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {avg_test_acc:.4f} ({(avg_test_acc*100):.2f}%)")

    # --- Classification Report ---
    print("\n--- Classification Report ---")
    if class_names_list:
        report = classification_report(all_labels_np, all_preds_np, target_names=class_names_list, zero_division=0)
        print(report)
    else:
        print("Warning: Class names not provided. Cannot print named classification report.")
        report = classification_report(all_labels_np, all_preds_np, zero_division=0)
        print(report)

    # --- Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels_np, all_preds_np)

    if class_names_list:
        cm_df = pd.DataFrame(cm, index=class_names_list, columns=class_names_list)
    else:
        cm_df = pd.DataFrame(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    if not os.path.exists(config.BASE_OUTPUT_DIR):
        os.makedirs(config.BASE_OUTPUT_DIR)
    plt.savefig(config.CONFUSION_MATRIX_SAVE_PATH)
    print(f"Confusion matrix saved to {config.CONFUSION_MATRIX_SAVE_PATH}")
    # plt.show() # Uncomment to display if your environment supports it

# --- Main Execution Block for Evaluation ---
if __name__ == '__main__':
    freeze_support()

    print(f"Using device: {config.DEVICE}")

    # 1. Get Data Transforms (we only need 'test' transforms here)
    data_transforms = utils.get_data_transforms()

    # 2. Create DataLoaders
    # We need test_loader, class_names, and num_classes for evaluation
    # The train_loader and val_loader from create_dataloaders will be None if not used,
    # but create_dataloaders is convenient for getting class_names and num_classes from the dataset structure.
    _, _, test_loader, class_names_from_data, num_classes_from_data = utils.create_dataloaders(
        data_transforms
    )

    if test_loader is None:
        print("Exiting due to dataloader creation errors. Check config.py and data paths.")
        exit()

    # Update config with dynamically determined class info if they weren't set
    # This ensures the model loaded matches the dataset structure.
    if config.NUM_CLASSES is None:
        config.NUM_CLASSES = num_classes_from_data
    if config.CLASS_NAMES is None: # Use the class names from the loaded dataset
        config.CLASS_NAMES = class_names_from_data


    # 3. Define the Model structure (same as used for training)
    if config.NUM_CLASSES is None:
        print("ERROR: NUM_CLASSES was not set. Cannot define model structure.")
        exit()
    model_pytorch = utils.get_model(num_classes=config.NUM_CLASSES)
    print(f"\nModel structure: {type(model_pytorch).__name__} defined for {config.NUM_CLASSES} output classes.")

    # 4. Load Trained Model Weights
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading trained model weights from: {config.MODEL_SAVE_PATH}")
        try:
            model_pytorch.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Ensure the model was saved correctly and config.NUM_CLASSES matches the saved model.")
            exit()
    else:
        print(f"ERROR: Trained model not found at {config.MODEL_SAVE_PATH}")
        print("Please train the model first using train.py or provide the correct path.")
        exit()

    # 5. Define Criterion (only needed if you want to calculate loss during evaluation)
    criterion = nn.CrossEntropyLoss()

    # 6. Evaluate the Model
    evaluate_model_on_test_set(model_pytorch, test_loader, criterion, config.CLASS_NAMES)

    print("\n--- Evaluation Script Finished ---")