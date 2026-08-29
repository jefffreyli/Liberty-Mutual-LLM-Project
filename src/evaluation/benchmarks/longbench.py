"""LongBench's English passage retrieval task, which is the length axis: the same
find-the-supporting-passage decision the model was trained on, but over ten to thirty paragraphs
and thousands of tokens of context. Exactly one paragraph is gold, so citation F1 is the metric
that matters here and answer coverage only checks that the response names the paragraph it cited.
"""

from __future__ import annotations

import json
import re
import zipfile

from huggingface_hub import hf_hub_download

from src.evaluation.benchmarks.base import Benchmark, build_pool
from src.evaluation.judge import GroundingVerdict
from src.schema import DecompositionStep, TrainingRow

DATASET_NAME = "zai-org/LongBench"
DATA_FILE = "data.zip"
# The length stratified variant, which is already 300 rows spread across short,
# medium, and long contexts.
TASK_FILE = "passage_retrieval_en_e.jsonl"
_PARAGRAPH_PATTERN = re.compile(r"Paragraph (\d+):\s*")
# Context length boundaries the rows are stratified over.
LENGTH_BUCKETS = (4000, 8000)


class LongBench(Benchmark):
    """LongBench passage retrieval rows, adapted into the answer format."""

    name = "longbench"
    verdict_model = GroundingVerdict
    reference_label = "the gold paragraph"
    # A thirty paragraph pool plus the usual three exemplars runs well past the
    # 4096 tokens the adapter was trained at, so this benchmark prompts with one.
    few_shot_override = 1

    def build_rows(self) -> list[TrainingRow]:
        """Read the task file out of the dataset archive and map each query onto a row.

        Returns:
            One row per query whose gold paragraph is present in its context.
        """
        archive_path = hf_hub_download(DATASET_NAME, DATA_FILE, repo_type="dataset")
        with zipfile.ZipFile(archive_path) as archive:
            member = next(name for name in archive.namelist() if name.endswith(TASK_FILE))
            with archive.open(member) as f:
                items = [json.loads(line) for line in f if line.strip()]

        rows = [_to_row(index, item) for index, item in enumerate(items)]
        return [row for row in rows if row is not None]


def _length_bucket(length: int) -> str:
    """Name the context length band a row falls in.

    Params:
        length: The row's reported context length.

    Returns:
        The bucket label used to stratify the sample.
    """
    if length < LENGTH_BUCKETS[0]:
        return "short"
    return "medium" if length < LENGTH_BUCKETS[1] else "long"


def _to_row(index: int, item: dict) -> TrainingRow | None:
    """Map one passage retrieval query onto a training row.

    Params:
        index: Position in the task file, used to build a stable row id.
        item: A raw task row, whose context is numbered paragraphs in one string
            and whose answer names the gold paragraph.

    Returns:
        The adapted row, or None if the answer names a paragraph the context
        does not contain.
    """
    pieces = _PARAGRAPH_PATTERN.split(item["context"])[1:]
    numbers, bodies = pieces[0::2], pieces[1::2]
    answer = item["answers"][0]
    gold_number = answer.split()[-1]
    if gold_number not in numbers:
        return None

    pool = build_pool(
        [
            (f"Paragraph {number}", body.strip(), number == gold_number)
            for number, body in zip(numbers, bodies)
        ]
    )
    gold_id = next(chunk.id for chunk in pool if chunk.is_informative)
    return TrainingRow(
        id=f"longbench_{index:04d}",
        source=f"longbench/{_length_bucket(item['length'])}",
        instruction=item["input"],
        decomposition=[
            DecompositionStep(
                id=0, instruction=item["input"], answer=answer, support_paragraph_id=gold_id
            )
        ],
        search_pool=pool,
        rationale="",
        response=answer,
    )
