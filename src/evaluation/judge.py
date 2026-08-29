"""Grades model responses with a strong LLM judge, formatting each response and the ground truth it
should have used into the rubric prompt and returning per metric scores with justifications. Two
verdict shapes exist because hop completeness needs a real per hop decomposition: benchmarks that
ship one are graded on all five metrics, and the rest on the four that need only the search pool
and the gold IDs.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

from pydantic import BaseModel

from src.config.models import JUDGE_MODELS, JUDGE_WORKERS
from src.evaluation.prompts import (
    DECOMPOSITION_SECTION,
    HOP_COMPLETENESS_RUBRIC,
    RESPONSE_JUDGE_PROMPT,
)
from src.llm.client import get_llm_client
from src.render import render_labeled_pool
from src.schema import MetricScore, TrainingRow
from src.training.format import ParsedAnswer, gold_informative_ids


class GroundingVerdict(BaseModel):
    """The judge's scores for one response, over the metrics every benchmark can support.

    `METRICS` names the fields in the order they appear in the prompt, so a
    subclass that adds a metric is aggregated without branching anywhere else.
    """

    METRICS: ClassVar[tuple[str, ...]] = (
        "chunk_selection",
        "rationale_quality",
        "answer_grounding",
        "distractor_resistance",
    )

    chunk_selection: MetricScore
    rationale_quality: MetricScore
    answer_grounding: MetricScore
    distractor_resistance: MetricScore

    def as_metrics(self) -> dict[str, float]:
        """Flatten the verdict for aggregation.

        Returns:
            Each metric's score plus judge_score, their mean.
        """
        scores = {metric: float(getattr(self, metric).score) for metric in type(self).METRICS}
        return {**scores, "judge_score": sum(scores.values()) / len(scores)}

    def justifications(self) -> dict[str, str]:
        """Collect the judge's reasoning.

        Returns:
            Metric name to justification.
        """
        return {metric: getattr(self, metric).justification for metric in type(self).METRICS}


class ResponseVerdict(GroundingVerdict):
    """A grounding verdict plus hop completeness, for rows with a real decomposition."""

    METRICS: ClassVar[tuple[str, ...]] = GroundingVerdict.METRICS + ("hop_completeness",)

    hop_completeness: MetricScore


class ResponseJudge:
    """Scores responses against their source rows using an LLM judge."""

    def __init__(
        self,
        model: str = JUDGE_MODELS[0],
        verdict_model: type[GroundingVerdict] = ResponseVerdict,
        reference_label: str = "written from the informative paragraphs only",
    ):
        self.model = model
        self.verdict_model = verdict_model
        self.reference_label = reference_label

    def grade(self, row: TrainingRow, parsed: ParsedAnswer) -> GroundingVerdict:
        """Grade one response.

        Params:
            row: The row the response was generated from.
            parsed: The model's answer, already split into its parts.

        Returns:
            The judge's verdict, in this judge's verdict shape.
        """
        prompt = self._build_prompt(row, parsed)
        return get_llm_client(self.model).generate(prompt, self.verdict_model)

    def grade_many(
        self,
        graded: list[tuple[TrainingRow, ParsedAnswer]],
        workers: int = JUDGE_WORKERS,
    ) -> list[GroundingVerdict | None]:
        """Grade responses in parallel.

        Params:
            graded: Row and parsed answer pairs to judge.
            workers: Parallel judge calls.

        Returns:
            One verdict per pair in the input order, or None where the judge
            failed, so one bad call cannot lose a whole evaluation run.
        """

        def grade_one(pair: tuple[TrainingRow, ParsedAnswer]) -> GroundingVerdict | None:
            try:
                return self.grade(*pair)
            except Exception as error:
                print(f"  {self.model} judge failed on row {pair[0].id}: {error}")
                return None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(grade_one, graded))

    def _build_prompt(self, row: TrainingRow, parsed: ParsedAnswer) -> str:
        """Format one response and its ground truth into the rubric prompt.

        The decomposition and its rubric metric are included only when this
        judge grades hop completeness, so a dataset without per hop answers is
        never asked to score against ground truth it does not have.

        Params:
            row: The row the response was generated from.
            parsed: The model's answer.

        Returns:
            The prompt text.
        """
        grades_hops = "hop_completeness" in self.verdict_model.METRICS
        decomposition_section = ""
        if grades_hops:
            decomposition_text = "\n".join(
                f"  Step {step.id}: {step.instruction} -> {step.answer}"
                for step in row.decomposition
            )
            decomposition_section = DECOMPOSITION_SECTION.format(
                decomposition_text=decomposition_text
            )

        return RESPONSE_JUDGE_PROMPT.format(
            instruction=row.instruction,
            search_pool_text=render_labeled_pool(row.search_pool),
            informative_ids=gold_informative_ids(row),
            decomposition_section=decomposition_section,
            hop_rubric=HOP_COMPLETENESS_RUBRIC if grades_hops else "",
            reference_label=self.reference_label,
            reference_response=row.response,
            cited_ids="(none parsed)" if parsed.cited_ids is None else parsed.cited_ids,
            rationale=parsed.rationale or "(none)",
            response=parsed.response or "(none)",
        )
