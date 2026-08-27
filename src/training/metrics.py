"""Grades a model answer against its training row, combining citation F1 over the gold search
result IDs, token coverage of the per-hop answers, and a penalty for vocabulary that only appears
in distractor chunks. RL optimizes this score directly and src/evaluation reports it, so the two
stages are always measuring the same thing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.schema import TrainingRow
from src.training.format import ParsedAnswer, gold_informative_ids, parse_answer

_WORD_PATTERN = re.compile(r"[a-z0-9$%.]+")

# Closed-class words carry no grounding signal, so any fluent response would
# otherwise earn coverage points for free.
_STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have he her his i if in into is it
    its of on or she that the their them there these they this to was were what when which who
    will with would you your""".split()
)


def _content_tokens(text: str) -> set[str]:
    """Tokenize text into lowercased content words.

    Params:
        text: Arbitrary text.

    Returns:
        The set of content tokens.
    """
    tokens = (token.strip(".") for token in _WORD_PATTERN.findall(text.lower()))
    return {token for token in tokens if token and token not in _STOPWORDS}


@dataclass(frozen=True)
class Grade:
    """A scored answer, where `score` is the RL reward and the rest are monitoring metrics."""

    score: float
    format_ok: bool
    citation_precision: float
    citation_recall: float
    citation_f1: float
    answer_coverage: float
    leakage: float

    def as_metrics(self) -> dict[str, float]:
        """Flatten the grade for the training metrics log.

        Returns:
            All fields as floats.
        """
        return {key: float(value) for key, value in asdict(self).items()}


def citation_scores(cited_ids: list[int], gold_ids: list[int]) -> tuple[float, float, float]:
    """Score cited search result IDs against the gold set.

    Params:
        cited_ids: IDs the model claimed were informative.
        gold_ids: IDs actually marked informative.

    Returns:
        Precision, recall, and F1.
    """
    cited, gold = set(cited_ids), set(gold_ids)
    if not gold:
        # With no gold chunks, citing nothing is the correct behavior.
        return (1.0, 1.0, 1.0) if not cited else (0.0, 1.0, 0.0)
    if not cited:
        return 0.0, 0.0, 0.0

    hits = len(cited & gold)
    precision = hits / len(cited)
    recall = hits / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return precision, recall, f1


def answer_coverage(response: str, row: TrainingRow) -> float:
    """Measure how much of each decomposition step's answer appears in the response.

    Params:
        response: The response section of the model answer.
        row: The training row holding the gold decomposition.

    Returns:
        Mean per-hop token recall in [0, 1]. Averaging per hop rather than
        pooling keeps a long final hop from hiding a missed intermediate one.
        An unanswerable row has no hops to cover, so it scores 1.0, matching how
        citation_scores treats an empty gold set.
    """
    if not row.decomposition:
        return 1.0

    response_tokens = _content_tokens(response)
    if not response_tokens:
        return 0.0

    recalls = []
    for step in row.decomposition:
        gold_tokens = _content_tokens(step.answer)
        if gold_tokens:
            recalls.append(len(gold_tokens & response_tokens) / len(gold_tokens))
    return sum(recalls) / len(recalls) if recalls else 0.0


def distractor_leakage(response: str, row: TrainingRow) -> float:
    """Measure how much of the response comes from distractor-only vocabulary.

    Params:
        response: The response section of the model answer.
        row: The training row holding the search pool.

    Returns:
        Fraction of response tokens in [0, 1] that appear only in distractor
        chunks. Normalizing by response length keeps the penalty stable as the
        pool grows.
    """
    response_tokens = _content_tokens(response)
    if not response_tokens:
        return 0.0

    informative_tokens: set[str] = set()
    distractor_tokens: set[str] = set()
    for chunk in row.search_pool:
        chunk_tokens = _content_tokens(f"{chunk.title} {chunk.text}")
        if chunk.is_informative:
            informative_tokens |= chunk_tokens
        else:
            distractor_tokens |= chunk_tokens

    # Words shared with the gold chunks or the instruction are fair game.
    exclusive = distractor_tokens - informative_tokens - _content_tokens(row.instruction)
    return len(response_tokens & exclusive) / len(response_tokens)


def grade_answer(
    parsed: ParsedAnswer,
    row: TrainingRow,
    *,
    citation_weight: float,
    answer_weight: float,
    leakage_weight: float,
) -> Grade:
    """Score a parsed answer against its row.

    Params:
        parsed: The parsed model answer.
        row: The training row it was generated from.
        citation_weight: Weight on citation F1.
        answer_weight: Weight on per-hop answer coverage.
        leakage_weight: Weight subtracted for distractor leakage.

    Returns:
        The grade, with `score` clipped to [0, 1]. A malformed answer scores 0;
        the format penalty itself is applied by the environment.
    """
    if not parsed.is_well_formed:
        return Grade(0.0, False, 0.0, 0.0, 0.0, 0.0, 0.0)

    precision, recall, f1 = citation_scores(parsed.cited_ids or [], gold_informative_ids(row))
    coverage = answer_coverage(parsed.response, row)
    leakage = distractor_leakage(parsed.response, row)
    score = citation_weight * f1 + answer_weight * coverage - leakage_weight * leakage
    return Grade(
        score=min(1.0, max(0.0, score)),
        format_ok=True,
        citation_precision=precision,
        citation_recall=recall,
        citation_f1=f1,
        answer_coverage=coverage,
        leakage=leakage,
    )


def grade_completion(completion: str, row: TrainingRow, **weights: float) -> Grade:
    """Parse and grade a raw model completion.

    Params:
        completion: Raw model output.
        row: The training row it was generated from.
        weights: Forwarded to `grade_answer`.

    Returns:
        The grade.
    """
    return grade_answer(parse_answer(completion), row, **weights)
