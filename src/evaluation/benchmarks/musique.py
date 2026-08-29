"""MuSiQue's validation split, the closest external benchmark to the training distribution and
the only one that ships per hop answers, so it is the one dataset scored with the full judge
rubric. It also seeded a third of the generated data, which is why loading it asserts that the
seeds and the rows being scored do not overlap.
"""

from __future__ import annotations

import json

from datasets import load_dataset

from src.evaluation.benchmarks.base import Benchmark, build_pool
from src.paths import RUNS_DIR
from src.schema import DecompositionStep, TrainingRow

DATASET_NAME = "dgslibisey/MuSiQue"
SPLIT = "validation"
# The generated run seeded from MuSiQue. Seeds are drawn from the train split, so
# this is a guard against that changing, not a known overlap.
SEED_RUN_PATH = RUNS_DIR / "gen_musique_1000.json"


class MuSiQue(Benchmark):
    """MuSiQue-Ans validation rows, adapted into the answer format."""

    name = "musique"
    reference_label = "the gold multi-hop answer"

    def build_rows(self) -> list[TrainingRow]:
        """Load the validation split and map each question onto a row.

        Returns:
            One row per validation question.

        Raises:
            ValueError: If any question also seeded the generated training data.
        """
        items = list(load_dataset(DATASET_NAME, split=SPLIT))
        _check_not_seeded(items)
        return [_to_row(item) for item in items]


def _to_row(item: dict) -> TrainingRow:
    """Map one MuSiQue question onto a training row.

    Params:
        item: A raw validation row.

    Returns:
        The adapted row, with `source` carrying the hop count it is stratified by.
    """
    pool = build_pool(
        [
            (paragraph["title"], paragraph["paragraph_text"], paragraph["is_supporting"])
            for paragraph in item["paragraphs"]
        ]
    )
    decomposition = [
        DecompositionStep(
            id=index,
            instruction=step["question"],
            answer=step["answer"],
            # Paragraph order is preserved, so MuSiQue's index is the pool ID.
            support_paragraph_id=step.get("paragraph_support_idx") or -1,
        )
        for index, step in enumerate(item["question_decomposition"])
    ]
    return TrainingRow(
        id=item["id"],
        source=f"musique/{len(decomposition)}hop",
        instruction=item["question"],
        decomposition=decomposition,
        search_pool=pool,
        rationale="",
        response=item["answer"],
    )


def _check_not_seeded(items: list[dict]) -> None:
    """Assert none of the rows about to be scored seeded the training data.

    Params:
        items: The raw validation rows.

    Raises:
        ValueError: If a validation question appears as a seed in the run file.
    """
    if not SEED_RUN_PATH.exists():
        return
    with open(SEED_RUN_PATH) as f:
        seeds = {row.get("seed") for row in json.load(f)["data"]}
    overlap = {item["question"] for item in items} & seeds
    if overlap:
        raise ValueError(
            f"{len(overlap)} MuSiQue validation questions seeded {SEED_RUN_PATH.name}, "
            f"for example {next(iter(overlap))!r}"
        )
