"""Converts a generated run JSON into the conversation JSONL that Tinker's SFT loop reads, and
reports the token length distribution so the max sequence length can be set before training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.schema import TrainingRow
from src.training.format import row_to_messages


def load_rows(data_path: Path) -> list[TrainingRow]:
    """Load and validate the rows of a generated run JSON.

    Params:
        data_path: Path to a run JSON with a top level "data" key.

    Returns:
        The validated training rows.

    Raises:
        ValueError: If the file is not a run JSON.
    """
    with open(data_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"Expected a dict with a 'data' key at {data_path}")
    return [TrainingRow.model_validate(row) for row in payload["data"]]


def split_sizes(num_rows: int, test_fraction: float, val_fraction: float) -> tuple[int, int]:
    """Resolve fractional split sizes into row counts.

    Params:
        num_rows: Size of the whole dataset.
        test_fraction: Share reserved for src/evaluation.
        val_fraction: Share reserved for monitoring during training.

    Returns:
        The test and validation row counts.

    Raises:
        ValueError: If the two fractions leave no rows to train on.
    """
    test_size = round(num_rows * test_fraction)
    val_size = round(num_rows * val_fraction)
    if test_size + val_size >= num_rows:
        raise ValueError(
            f"test_fraction + val_fraction leave no training rows in {num_rows} rows"
        )
    return test_size, val_size


def split_rows(
    rows: list[TrainingRow], test_size: int, val_size: int, seed: int
) -> tuple[list[TrainingRow], list[TrainingRow], list[TrainingRow]]:
    """Split rows into train, validation, and held out test sets.

    Every trainer and src/evaluation call this with the same seed, so the test
    rows are the same everywhere and never leak into training.

    Params:
        rows: All rows loaded from a run JSON.
        test_size: How many rows to reserve for src/evaluation.
        val_size: How many rows to reserve for monitoring during training.
        seed: Shuffle seed.

    Returns:
        The train, validation, and test rows.
    """
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    test = shuffled[:test_size]
    val = shuffled[test_size : test_size + val_size]
    return shuffled[test_size + val_size :], val, test


def write_conversations(rows: list[TrainingRow], output_path: Path) -> None:
    """Write rows as one JSON conversation per line.

    Params:
        rows: The training rows.
        output_path: Destination JSONL path, created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps({"messages": row_to_messages(row)}, ensure_ascii=False) + "\n")


def report_token_lengths(rows: list[TrainingRow], model_name: str, max_length: int) -> None:
    """Print the token length distribution of the rendered conversations.

    Examples longer than the training max length are silently dropped by the
    trainer, so this is the check that they are rare before a run starts.

    Params:
        rows: The training rows.
        model_name: Model whose tokenizer is used for counting.
        max_length: The max sequence length training will use.
    """
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    lengths = sorted(
        len(tokenizer.encode("\n".join(m["content"] for m in row_to_messages(row)))) for row in rows
    )

    def percentile(fraction: float) -> int:
        return lengths[min(len(lengths) - 1, int(fraction * len(lengths)))]

    over = sum(1 for length in lengths if length > max_length)
    print(
        f"Tokens per example: median={percentile(0.5)}, p95={percentile(0.95)}, "
        f"max={lengths[-1]}; {over}/{len(lengths)} exceed max_length={max_length}"
    )
