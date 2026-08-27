"""Entry point for GRPO reinforcement learning on top of the SFT checkpoint. Any field of
src/config/training.py can be overridden on the command line.

Run:
    python3 -m scripts.train_rl
    python3 -m scripts.train_rl learning_rate=1e-5 max_tokens=3072
"""

from src.training.rl import run

if __name__ == "__main__":
    run()
