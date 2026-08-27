"""Quality gate for generated rows: asks the teacher model to score a row against the rubric,
then converts those scores into a pass or fail decision using the weights, threshold, and hard
constraints in src/config/generation.py. Rows with no informative chunk are scored against their
own rubric, since necessity and grounding mean nothing without gold chunks.
"""

from src.config.generation import (
    HARD_PASS_METRICS,
    METRIC_WEIGHTS,
    QUALITY_THRESHOLD,
    UNANSWERABLE_HARD_PASS_METRICS,
    UNANSWERABLE_METRIC_WEIGHTS,
)
from src.generation.prompts import RUBRIC_PROMPT, UNANSWERABLE_RUBRIC_PROMPT
from src.render import render_labeled_pool, render_pool
from src.schema import EvaluationResult, RubricVerdict, TrainingRow, UnanswerableVerdict
from src.llm.client import get_llm_client


def _build_rubric_prompt(row: TrainingRow) -> str:
    """Format a row into the rubric evaluation prompt.

    Params:
        row: The candidate row.

    Returns:
        The prompt text.
    """
    decomposition_text = "\n".join(
        f"  Step {s.id}: {s.instruction} -> {s.answer} "
        f"(support: paragraph {s.support_paragraph_id})"
        for s in row.decomposition
    )
    return RUBRIC_PROMPT.format(
        instruction=row.instruction,
        decomposition_text=decomposition_text,
        search_pool_text=render_labeled_pool(row.search_pool),
        informative_ids=[s.id for s in row.search_pool if s.is_informative],
        response=row.response,
    )


def _build_unanswerable_prompt(row: TrainingRow) -> str:
    """Format a row with no informative chunk into its rubric prompt.

    Params:
        row: The candidate row.

    Returns:
        The prompt text.
    """
    return UNANSWERABLE_RUBRIC_PROMPT.format(
        instruction=row.instruction,
        search_pool_text=render_pool(row.search_pool),
        response=row.response,
    )


def _metric_scores(
    verdict: RubricVerdict | UnanswerableVerdict, weights: dict[str, float]
) -> dict[str, int]:
    """Read the scored metrics named by the config off the verdict.

    Params:
        verdict: The rubric verdict.
        weights: Metric weights naming the metrics to read.

    Returns:
        Metric name to score.

    Raises:
        AttributeError: If the config names a metric the verdict schema lacks,
            which means the two drifted apart.
    """
    return {metric: getattr(verdict, metric).score for metric in weights}


def _decide(
    verdict: RubricVerdict | UnanswerableVerdict,
    weights: dict[str, float],
    hard_pass: list[str],
) -> EvaluationResult:
    """Apply the weighted threshold and the hard constraints.

    Params:
        verdict: The rubric verdict.
        weights: Metric weights that sum to one.
        hard_pass: Metrics that reject the row outright when they score zero.

    Returns:
        The gate decision, listing every reason it failed.
    """
    scores = _metric_scores(verdict, weights)
    aggregate = sum(scores[metric] * weight for metric, weight in weights.items())

    failure_reasons = [
        f"{metric} failed (hard constraint)" for metric in hard_pass if scores[metric] == 0
    ]
    if aggregate < QUALITY_THRESHOLD:
        failure_reasons.append(
            f"aggregate score {aggregate:.2f} below threshold {QUALITY_THRESHOLD}"
        )

    return EvaluationResult(
        verdict=verdict,
        aggregate_score=aggregate,
        passed=not failure_reasons,
        failure_reasons=failure_reasons,
    )


def evaluate_row(row: TrainingRow) -> EvaluationResult:
    """Score a row against the rubric it belongs to and decide whether to keep it.

    Params:
        row: The candidate row.

    Returns:
        The gate decision.
    """
    if not any(chunk.is_informative for chunk in row.search_pool):
        verdict = get_llm_client().generate(
            _build_unanswerable_prompt(row), UnanswerableVerdict
        )
        return _decide(verdict, UNANSWERABLE_METRIC_WEIGHTS, UNANSWERABLE_HARD_PASS_METRICS)

    verdict = get_llm_client().generate(_build_rubric_prompt(row), RubricVerdict)
    return _decide(verdict, METRIC_WEIGHTS, HARD_PASS_METRICS)
