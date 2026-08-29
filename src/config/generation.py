"""Knobs for the synthetic data generation pipeline: how many rows to build, how much noise each
one carries, what share of them are unanswerable, and how strict the rubric gates are before a
row is kept.
"""

# Dataset
NUM_ROWS = 1000
NUM_WORKERS = 50  # parallel threads for row generation (each gets its own LLM client)
NUM_DISTRACTORS = 8
# Chunks that state a false version of a gold fact. These add veracity
# discrimination on top of relevance, which is the half of the task plain
# distractors cannot test. Set to 0 to generate rows without them.
NUM_CONTRADICTORY = 2
MAX_RETRIES_PER_ROW = 10

# A row is rejected when ranking chunks by word overlap with the instruction
# recovers more than this much of the gold set. The instruction and its gold
# paragraphs are written in one teacher call, so without this gate they share
# vocabulary and the task is partly solvable without reading. Measured on the
# previous dataset, 0.67 would have rejected 34% of rows and 1.0 would have
# rejected the 26% that were exactly solvable.
LEXICAL_SHORTCUT_MAX_F1 = 0.67

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
    "logical_necessity": 0.2,
    "distractor_plausibility": 0.2,
    "non_contradiction": 0.2,
    "answer_grounding": 0.2,
    "contradiction_validity": 0.2,
}
HARD_PASS_METRICS = [
    "non_contradiction",
    "answer_grounding",
    "logical_necessity",
    "contradiction_validity",
]

# Rubric gate for unanswerable rows. Scored the same way against its own metrics,
# since logical necessity and answer grounding are meaningless with no gold chunks.
UNANSWERABLE_METRIC_WEIGHTS = {
    "no_support": 0.4,
    "abstention_correctness": 0.4,
    "distractor_plausibility": 0.2,
}
UNANSWERABLE_HARD_PASS_METRICS = ["no_support", "abstention_correctness"]
