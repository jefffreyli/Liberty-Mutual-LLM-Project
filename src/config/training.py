"""Configuration for Tinker fine-tuning (SFT + RL).

Every knob both training entry points read lives here. Anything in this file can
also be overridden on the command line, because the entry points hand their
config to `chz`:

    python3 -m scripts.train_sft learning_rate=1e-4 num_epochs=2
    python3 -m scripts.train_rl group_size=16 max_tokens=3072
"""

from src.paths import CHECKPOINTS_DIR, RUNS_DIR

# Data
DATA_PATH = RUNS_DIR / "combined_3750.json"
SFT_JSONL_PATH = RUNS_DIR / "sft" / "conversations_3750.jsonl"
# A 90/5/5 train, validation, test split. Fractions rather than counts so the
# ratio holds as the dataset grows. Test rows are reserved for src/evaluation and
# neither trainer ever sees them, so scores stay comparable across the base
# model, SFT, and RL. Validation rows are for monitoring during a run.
TEST_FRACTION = 0.05
VAL_FRACTION = 0.05
SEED = 42

# Model. Qwen3.5 is a hybrid thinking model, and the two renderers it supports
# differ in what the assistant turn looks like: `qwen3_5` leaves a `<think>`
# block open for the model to fill, `qwen3_5_disable_thinking` prefills an empty
# one. Our targets are teacher rationales, not chain-of-thought traces, so we
# train and sample in non-thinking mode. Set RENDERER_NAME to None to fall back
# to the model's default recommended renderer.
MODEL_NAME = "Qwen/Qwen3.5-9B"
RENDERER_NAME = "qwen3_5_disable_thinking"
LORA_RANK = 32

# Supervised fine-tuning
SFT_LOG_PATH = CHECKPOINTS_DIR / "sft"
SFT_LEARNING_RATE = 1e-4
SFT_LR_SCHEDULE = "linear"
SFT_NUM_EPOCHS = 2
SFT_BATCH_SIZE = 16  # rows per batch
SFT_MAX_LENGTH = 4096  # tokens per example; the data tops out near 3k
SFT_SAVE_EVERY = 20
SFT_EVAL_EVERY = 10

# Reinforcement learning (GRPO). `groups_per_batch` instructions per step, each
# sampled `group_size` times; advantages are centered within a group.
RL_LOG_PATH = CHECKPOINTS_DIR / "rl"
RL_LEARNING_RATE = 4e-5
RL_GROUP_SIZE = 8
RL_GROUPS_PER_BATCH = 16
RL_MAX_TOKENS = 2048  # generated tokens per rollout
RL_KL_PENALTY_COEF = 0.0
RL_SAVE_EVERY = 20
RL_EVAL_EVERY = 20
# Start RL from the SFT run's final checkpoint. None trains RL from the base
# model, which is a much weaker starting point for this format.
RL_INIT_FROM_SFT = True

# Reward shaping (see src/training/metrics.py). The three weighted terms should
# sum to roughly 1 so the total reward stays in [0, 1] before the format bonus.
REWARD_CITATION_WEIGHT = 0.5  # F1 of cited informative IDs against the gold set
REWARD_ANSWER_WEIGHT = 0.5  # coverage of the gold per-hop answers
REWARD_LEAKAGE_WEIGHT = 0.2  # penalty for parroting distractor-only vocabulary
REWARD_FORMAT_COEF = 0.1  # penalty applied when the answer format is unparseable

# Weights & Biases. Every metric is written to metrics.jsonl under the run's log
# path regardless, so a run without WANDB_API_KEY set loses only the dashboard.
# Set WANDB_PROJECT to None to turn the upload off outright.
WANDB_PROJECT = "multihop-instruction-following"

# Where each run writes its metrics and checkpoints, keyed by the name that the
# evaluation and export entry points take on the command line.
RUN_LOG_PATHS = {"sft": SFT_LOG_PATH, "rl": RL_LOG_PATH}

# Weight export
EXPORT_DIR = CHECKPOINTS_DIR / "export"
HF_REPO_ID = "jefffreyli/multihop-qwen3.5-9b"
