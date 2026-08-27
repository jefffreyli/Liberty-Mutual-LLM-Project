"""Schemas for rubric evaluation and gate decisions."""

from pydantic import BaseModel


class MetricScore(BaseModel):
    """Single rubric metric evaluation result."""

    score: int
    justification: str


class RubricVerdict(BaseModel):
    """
    LLM output: rubric scores for a training row across quality metrics.

    Logical necessity: Every informative paragraph is required to complete the instruction. Removing any single informative paragraph would make the instruction unsatisfiable.
    Distractor plausibility: Distractor paragraphs are topically related to the instruction and share high semantic similarity with the task domain. They should be genuinely plausible retrieval results a search engine might return.
    Non-contradiction: No distractor paragraph contains false versions of facts stated in the informative paragraphs. Distractors should cover neighboring concepts (different jurisdictions, policy types, time periods) rather than contradicting the truth.
    Answer grounding: The response is fully supported by and grounded in ONLY the informative paragraphs. It contains no hallucinated facts or outside knowledge.
    """

    logical_necessity: MetricScore
    distractor_plausibility: MetricScore
    non_contradiction: MetricScore
    answer_grounding: MetricScore


class UnanswerableVerdict(BaseModel):
    """
    LLM output: rubric scores for a row whose search pool holds no informative chunk.

    No support: No search result, alone or combined with the others, provides the information needed to complete the instruction.
    Distractor plausibility: The search results are topically related to the instruction and are plausible retrieval results a search engine might return.
    Abstention correctness: The response declines to answer, names the missing information, and does not fabricate or import outside knowledge.
    """

    no_support: MetricScore
    distractor_plausibility: MetricScore
    abstention_correctness: MetricScore


class EvaluationResult(BaseModel):
    """Computed evaluation outcome with gate decision."""

    verdict: RubricVerdict | UnanswerableVerdict
    aggregate_score: float
    passed: bool
    failure_reasons: list[str]
