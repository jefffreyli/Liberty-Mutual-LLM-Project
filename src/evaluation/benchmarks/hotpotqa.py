"""HotpotQA's distractor validation split: ten paragraphs, two of them supporting, with distractors
drawn by a retriever rather than written by a teacher. That different noise is the point of
including it. It carries no intermediate answers, so its decomposition is a single step holding the
final answer and it is scored with the four metric judge rubric.
"""

from __future__ import annotations

from datasets import load_dataset

from src.evaluation.benchmarks.base import Benchmark, build_pool
from src.evaluation.judge import GroundingVerdict
from src.schema import DecompositionStep, TrainingRow

DATASET_NAME = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"


class HotpotQA(Benchmark):
    """HotpotQA distractor validation rows, adapted into the answer format."""

    name = "hotpotqa"
    verdict_model = GroundingVerdict
    reference_label = "the gold answer span"

    def build_rows(self) -> list[TrainingRow]:
        """Load the distractor validation split and map each question onto a row.

        Returns:
            One row per validation question.
        """
        items = load_dataset(DATASET_NAME, CONFIG, split=SPLIT)
        return [_to_row(item) for item in items]


def _to_row(item: dict) -> TrainingRow:
    """Map one HotpotQA question onto a training row.

    Params:
        item: A raw validation row, whose context is parallel title and sentence
            lists and whose supporting facts name titles rather than indices.

    Returns:
        The adapted row, with `source` carrying the type and level it is
        stratified by.
    """
    supporting = set(item["supporting_facts"]["title"])
    titles = item["context"]["title"]
    pool = build_pool(
        [
            (title, "".join(sentences), title in supporting)
            for title, sentences in zip(titles, item["context"]["sentences"])
        ]
    )
    first_gold = next((chunk.id for chunk in pool if chunk.is_informative), -1)
    return TrainingRow(
        id=item["id"],
        source=f"hotpotqa/{item['type']}/{item['level']}",
        instruction=item["question"],
        # No intermediate answers exist, so the whole question is one step.
        decomposition=[
            DecompositionStep(
                id=0,
                instruction=item["question"],
                answer=item["answer"],
                support_paragraph_id=first_gold,
            )
        ],
        search_pool=pool,
        rationale="",
        response=item["answer"],
    )
