"""Entry point for supervised fine-tuning. Any field of src/config/training.py can be overridden
on the command line, because the config is handed to `chz` before the run starts.

Run:
    python3 -m scripts.train_sft
    python3 -m scripts.train_sft learning_rate=2e-4 num_epochs=1
"""

from src.training.sft import run

if __name__ == "__main__":
    run()
