"""Rubric-based evaluation gate for SAIL training rows."""

from ..config import HARD_PASS_METRICS, METRIC_WEIGHTS, QUALITY_THRESHOLD
from ..schema import EvaluationResult, RubricVerdict, TrainingRow
from ..utils.llm_client import generate
from .prompts import RUBRIC_PROMPT


def _build_rubric_prompt(row: TrainingRow) -> str:
    """Format a TrainingRow into the rubric evaluation prompt."""
    decomposition_text = "\n".join(
        f"  Step {s.id}: {s.instruction} -> {s.answer} "
        f"(support: paragraph {s.support_paragraph_id})"
        for s in row.decomposition
    )
    search_pool_text = "\n\n".join(
        f"[ID {s.id}] [{s.title}] (informative={s.is_informative}): {s.text}"
        for s in row.search_pool
    )
    informative_ids = [s.id for s in row.search_pool if s.is_informative]

    return RUBRIC_PROMPT.format(
        instruction=row.instruction,
        decomposition_text=decomposition_text,
        search_pool_text=search_pool_text,
        informative_ids=informative_ids,
        response=row.response,
    )


def _compute_gate_decision(verdict: RubricVerdict) -> EvaluationResult:
    """Apply weighted threshold and hard-pass constraints to produce a gate decision."""
    scores = {
        "logical_necessity": verdict.logical_necessity.score,
        "distractor_plausibility": verdict.distractor_plausibility.score,
        "non_contradiction": verdict.non_contradiction.score,
        "answer_grounding": verdict.answer_grounding.score,
    }

    aggregate = sum(scores[metric] * weight for metric, weight in METRIC_WEIGHTS.items())
    failure_reasons: list[str] = []

    for metric in HARD_PASS_METRICS:
        if scores[metric] == 0:
            failure_reasons.append(f"{metric} failed (hard constraint)")

    if aggregate < QUALITY_THRESHOLD:
        failure_reasons.append(
            f"aggregate score {aggregate:.2f} below threshold {QUALITY_THRESHOLD}"
        )

    return EvaluationResult(
        verdict=verdict,
        aggregate_score=aggregate,
        passed=len(failure_reasons) == 0,
        failure_reasons=failure_reasons,
    )


def evaluate_row(row: TrainingRow) -> EvaluationResult:
    """Evaluate a training row against the rubric and return a gate decision."""
    prompt = _build_rubric_prompt(row)
    verdict = generate(prompt, RubricVerdict)
    return _compute_gate_decision(verdict)
