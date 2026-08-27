"""Knobs for the synthetic data generation pipeline: how many rows to build, how much noise each
one carries, what share of them are unanswerable, and how strict the rubric gates are before a
row is kept.
"""

# Dataset
NUM_ROWS = 1000
NUM_WORKERS = 50  # parallel threads for row generation (each gets its own LLM client)
NUM_DISTRACTORS = 8
MAX_RETRIES_PER_ROW = 10

# Share of rows whose search pool holds no informative chunk at all, so the model
# sees pools where citing nothing is the correct answer. SAIL samples grounding
# this way; without it a discriminator learns to always cite a few chunks.
UNANSWERABLE_FRACTION = 0.2
NUM_UNANSWERABLE_DISTRACTORS = 11  # matches the mean pool size of answerable rows
SEED_DATASET_NAME = "alabnii/morehopqa"  # "dgslibisey/MuSiQue"

# Rubric gate. A row is kept when its weighted score clears the threshold and no
# hard pass metric scored zero.
QUALITY_THRESHOLD = 0.75
METRIC_WEIGHTS = {
    "logical_necessity": 0.25,
    "distractor_plausibility": 0.25,
    "non_contradiction": 0.25,
    "answer_grounding": 0.25,
}
HARD_PASS_METRICS = ["non_contradiction", "answer_grounding", "logical_necessity"]

# Rubric gate for unanswerable rows. Scored the same way against its own metrics,
# since logical necessity and answer grounding are meaningless with no gold chunks.
UNANSWERABLE_METRIC_WEIGHTS = {
    "no_support": 0.4,
    "abstention_correctness": 0.4,
    "distractor_plausibility": 0.2,
}
UNANSWERABLE_HARD_PASS_METRICS = ["no_support", "abstention_correctness"]
