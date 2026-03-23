"""Centralized configuration for the SAIL data generation pipeline."""

DEFAULT_MODEL = "gpt-4o"

NUM_DISTRACTORS = 8
MAX_RETRIES_PER_ROW = 3

QUALITY_THRESHOLD = 0.75

METRIC_WEIGHTS = {
    "logical_necessity": 0.25,
    "distractor_plausibility": 0.25,
    "non_contradiction": 0.25,
    "answer_grounding": 0.25,
}

HARD_PASS_METRICS = ["non_contradiction", "answer_grounding", "logical_necessity"]
