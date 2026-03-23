"""Schemas for rubric evaluation and gate decisions."""

from pydantic import BaseModel


class MetricScore(BaseModel):
    """Single rubric metric evaluation result."""

    score: int
    justification: str


class RubricVerdict(BaseModel):
    """LLM output: rubric scores for a training row across quality metrics."""

    logical_necessity: MetricScore
    distractor_plausibility: MetricScore
    non_contradiction: MetricScore
    answer_grounding: MetricScore


class EvaluationResult(BaseModel):
    """Computed evaluation outcome with gate decision."""

    verdict: RubricVerdict
    aggregate_score: float
    passed: bool
    failure_reasons: list[str]
