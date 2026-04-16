"""Centralized configuration for the SAIL data generation pipeline."""

# Dataset
NUM_ROWS = 100
NUM_DISTRACTORS = 8
MAX_RETRIES_PER_ROW = 5
QUALITY_THRESHOLD = 0.75
METRIC_WEIGHTS = {
    "logical_necessity": 0.25,
    "distractor_plausibility": 0.25,
    "non_contradiction": 0.25,
    "answer_grounding": 0.25,
}
HARD_PASS_METRICS = ["non_contradiction", "answer_grounding", "logical_necessity"]
SEED_DATASET_NAME = "tatsu-lab/alpaca"

# Model
DEFAULT_MODEL = "gpt-4o"
TOKEN_PRICE = {
    "gpt-4o": {
        "input": 2.5,
        "output": 10.0,
    },
}