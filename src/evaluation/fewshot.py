"""Builds the prompt an evaluated model sees: the shared system turn, some worked exemplars, and
the row under test. Exemplars are drawn from the training split so that a prompted baseline is
never shown a row it is about to be scored on.
"""

from __future__ import annotations

import random

from src.config import training as train_cfg
from src.schema import TrainingRow
from src.training.data import load_rows, split_rows, split_sizes
from src.training.format import build_assistant_target, build_user_prompt


def _train_rows() -> list[TrainingRow]:
    """Load the split the trainers learn from.

    Returns:
        The training rows, which exclude everything src/evaluation scores.
    """
    rows = load_rows(train_cfg.DATA_PATH)
    test_size, val_size = split_sizes(
        len(rows), train_cfg.TEST_FRACTION, train_cfg.VAL_FRACTION
    )
    train_rows, _, _ = split_rows(rows, test_size, val_size, train_cfg.SEED)
    return train_rows


def select_exemplars(count: int, seed: int = train_cfg.SEED) -> list[TrainingRow]:
    """Pick the worked examples a prompted baseline is shown.

    One exemplar is always unanswerable once there is room for two, because a
    prompt whose every example cites chunks teaches the model to always cite
    something, and abstention is a fifth of the rows it is about to be scored on.

    Params:
        count: How many exemplars to select.
        seed: Shuffle seed, so a rerun shows the same examples in the same order.

    Returns:
        The exemplar rows, or an empty list when count is zero.

    Raises:
        ValueError: If the training split cannot supply that many exemplars of
            the required kinds.
    """
    if count <= 0:
        return []

    rows = _train_rows()
    answerable = [row for row in rows if row.is_answerable]
    unanswerable = [row for row in rows if not row.is_answerable]

    num_unanswerable = 1 if count >= 2 and unanswerable else 0
    num_answerable = count - num_unanswerable
    if len(answerable) < num_answerable or len(unanswerable) < num_unanswerable:
        raise ValueError(
            f"Cannot draw {count} exemplars from {len(answerable)} answerable and "
            f"{len(unanswerable)} unanswerable training rows"
        )

    rng = random.Random(seed)
    selected = rng.sample(answerable, num_answerable) + rng.sample(unanswerable, num_unanswerable)
    rng.shuffle(selected)
    return selected


def build_messages(row: TrainingRow, exemplars: list[TrainingRow]) -> list[dict[str, str]]:
    """Build the user and assistant turns for one row under test.

    The system turn is left out because the two provider APIs disagree on where
    it goes; every sampler pairs these messages with `format.SYSTEM_PROMPT`.

    Params:
        row: The row to answer.
        exemplars: Worked examples to show first, in order.

    Returns:
        Alternating user and assistant turns, ending on the user turn that asks
        about `row`.
    """
    messages: list[dict[str, str]] = []
    for exemplar in exemplars:
        messages.append({"role": "user", "content": build_user_prompt(exemplar)})
        messages.append({"role": "assistant", "content": build_assistant_target(exemplar)})
    messages.append({"role": "user", "content": build_user_prompt(row)})
    return messages
