# config.py
import os
import torch

# --- Project Root ---
# Assuming config.py is in the project root, or adjust as needed
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Data Directories ---
# IMPORTANT: MODIFY these to point to your *already split* dataset
# Example: if your dataset 'my_image_data' has 'train', 'val', 'test' subfolders
# BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "my_image_data") # Example
BASE_DATA_DIR = "/Users/damion/Documents/gemini1/outputs"  # MODIFY: Update this path
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
VAL_DIR = os.path.join(BASE_DATA_DIR, "validation")  # Or "validation"
TEST_DIR = os.path.join(BASE_DATA_DIR, "test")

# --- Output Directory ---
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pytorch_outputs_gemini1")
MODEL_SAVE_PATH = os.path.join(BASE_OUTPUT_DIR, "best_pytorch_model.pth")
CONFUSION_MATRIX_SAVE_PATH = os.path.join(BASE_OUTPUT_DIR, "confusion_matrix.png")

# --- Model and Training Hyperparameters ---
NUM_CLASSES = None  # Will be determined dynamically in utils.py by create_dataloaders
CLASS_NAMES = None  # Will be determined dynamically in utils.py

IMAGE_SIZE = 224    # Input image size for the model
BATCH_SIZE = 32
NUM_EPOCHS = 15     # MODIFY: Adjust as needed
LEARNING_RATE = 0.001
# IMPORTANT: Start with NUM_WORKERS = 0.
# If NUM_WORKERS > 0, the `if __name__ == '__main__':` block in train.py/evaluate.py is CRITICAL.
NUM_WORKERS = 0 # For macOS/Windows, start with 0. If stable, try 2 or 4.

# --- Device Selection ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else \
                      "cpu")

# --- Flags ---
# Set to True if you want to load a pretrained model from torchvision
PRETRAINED_MODEL = True