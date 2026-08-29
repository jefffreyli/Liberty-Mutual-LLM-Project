"""FinQA's test split, the furthest benchmark from the training distribution: financial filings
rather than encyclopedia prose, and a numeric answer reached by arithmetic over the cited
evidence. The pool is assembled from one report page, where `gold_inds` names the supporting
sentences and table rows as `text_N` and `table_N`.
"""

from __future__ import annotations

import json

from huggingface_hub import hf_hub_download

from src.evaluation.benchmarks.base import Benchmark, build_pool
from src.evaluation.judge import GroundingVerdict
from src.schema import DecompositionStep, TrainingRow

DATASET_NAME = "TableQAKit/FinQA_retrieval"
DATA_FILE = "test.json"
# Titles are constant per kind so that no index word ends up exclusive to a
# distractor chunk, which would make distractor_leakage penalise a response for
# naming a figure that happens to share a chunk index.
TEXT_TITLE = "Text"
TABLE_TITLE = "Table row"


class FinQA(Benchmark):
    """FinQA test rows, adapted into the answer format."""

    name = "finqa"
    verdict_model = GroundingVerdict
    reference_label = "the gold numeric result"

    def build_rows(self) -> list[TrainingRow]:
        """Download the test split and map each question onto a row.

        Returns:
            One row per question that names its supporting evidence.
        """
        local_path = hf_hub_download(DATASET_NAME, DATA_FILE, repo_type="dataset")
        with open(local_path) as f:
            items = json.load(f)
        return [_to_row(item) for item in items if item["qa"].get("gold_inds")]


def _to_row(item: dict) -> TrainingRow:
    """Map one FinQA question onto a training row.

    The pool is the page's sentences followed by its table rows, so a `text_N`
    key is pool ID N and a `table_N` key is pool ID N offset past the sentences.

    Params:
        item: A raw test row.

    Returns:
        The adapted row.
    """
    sentences = item["pre_text"] + item["post_text"]
    gold = item["qa"]["gold_inds"]
    gold_text = {int(key.split("_")[1]) for key in gold if key.startswith("text_")}
    gold_table = {int(key.split("_")[1]) for key in gold if key.startswith("table_")}

    chunks = [(TEXT_TITLE, text, index in gold_text) for index, text in enumerate(sentences)]
    chunks += [
        (TABLE_TITLE, " | ".join(cells), index in gold_table)
        for index, cells in enumerate(item["table"])
    ]

    answer = str(item["qa"]["answer"])
    return TrainingRow(
        id=item["id"],
        source="finqa",
        instruction=item["qa"]["question"],
        # The arithmetic is one step: the program is not shown, only its result.
        decomposition=[
            DecompositionStep(
                id=0,
                instruction=item["qa"]["question"],
                answer=answer,
                support_paragraph_id=min(gold_text | {len(sentences) + i for i in gold_table}),
            )
        ],
        search_pool=build_pool(chunks),
        rationale="",
        response=answer,
    )
