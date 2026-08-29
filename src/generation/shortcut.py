"""Rejects rows whose gold chunks can be found by word overlap alone. The instruction and its
informative paragraphs are written in one teacher call, so they share vocabulary and a bag of
words retriever can recover the gold set without reasoning. This gate scores each candidate the
way that retriever would and throws the row away when it succeeds.
"""

from src.schema import TrainingRow
from src.text import content_tokens


def lexical_ranking_f1(row: TrainingRow) -> float:
    """Score how well word overlap alone recovers the row's gold chunks.

    Ranks every chunk by the share of instruction content words it contains and
    keeps as many as the row has gold chunks, which is the strongest form of the
    shortcut since it hands the retriever the right count for free.

    Params:
        row: The candidate row.

    Returns:
        F1 of the recovered set against the gold set, in [0, 1]. Returns 0.0 for
        a row with no gold chunks, where there is no shortcut to take.
    """
    gold_ids = {chunk.id for chunk in row.search_pool if chunk.is_informative}
    if not gold_ids:
        return 0.0

    instruction_tokens = content_tokens(row.instruction)
    if not instruction_tokens:
        return 0.0

    overlap = [
        (
            len(instruction_tokens & content_tokens(f"{chunk.title} {chunk.text}"))
            / len(instruction_tokens),
            chunk.id,
        )
        for chunk in row.search_pool
    ]
    picked = {chunk_id for _, chunk_id in sorted(overlap, reverse=True)[: len(gold_ids)]}

    hits = len(picked & gold_ids)
    if not hits:
        return 0.0
    precision = hits / len(picked)
    recall = hits / len(gold_ids)
    return 2 * precision * recall / (precision + recall)


def is_lexically_solvable(row: TrainingRow, max_f1: float) -> bool:
    """Decide whether word overlap alone recovers too much of the gold set.

    Params:
        row: The candidate row.
        max_f1: Highest lexical F1 a row may score and still be kept.

    Returns:
        True when the row should be rejected and regenerated.
    """
    return lexical_ranking_f1(row) > max_f1
