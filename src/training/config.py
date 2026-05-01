"""Training configuration for the SAIL SFT pipeline."""

from pathlib import Path

# Data
DATA_PATH = Path("runs/generated_1000.json")
EVAL_FRACTION = 0.1  # fraction of data held out for evaluation (0 to disable)
SEED = 42

# Model
MODEL_NAME = "Qwen/Qwen3.5-9B"
OUTPUT_DIR = Path("checkpoints/qwen3_5_9b_sft")

# LoRA
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
NUM_EPOCHS = 3.0
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
MAX_LENGTH = 8192
BF16 = True
