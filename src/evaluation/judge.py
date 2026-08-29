"""Grades model responses with a strong LLM judge, formatting each response and the ground truth
it should have used into the rubric prompt and returning per metric scores with justifications.
"""

from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from src.config.models import JUDGE_MODELS, JUDGE_WORKERS
from src.evaluation.prompts import RESPONSE_JUDGE_PROMPT
from src.render import render_labeled_pool
from src.schema import MetricScore, TrainingRow
from src.training.format import ParsedAnswer, gold_informative_ids
from src.llm.client import get_llm_client

# Rubric metrics, in the order they appear in the prompt.
JUDGE_METRICS = (
    "chunk_selection",
    "rationale_quality",
    "hop_completeness",
    "answer_grounding",
    "distractor_resistance",
)


class ResponseVerdict(BaseModel):
    """The judge's scores for one response, one MetricScore per rubric metric."""

    chunk_selection: MetricScore
    rationale_quality: MetricScore
    hop_completeness: MetricScore
    answer_grounding: MetricScore
    distractor_resistance: MetricScore

    def as_metrics(self) -> dict[str, float]:
        """Flatten the verdict for aggregation.

        Returns:
            Each metric's score plus judge_score, their mean.
        """
        scores = {metric: float(getattr(self, metric).score) for metric in JUDGE_METRICS}
        return {**scores, "judge_score": sum(scores.values()) / len(scores)}

    def justifications(self) -> dict[str, str]:
        """Collect the judge's reasoning.

        Returns:
            Metric name to justification.
        """
        return {metric: getattr(self, metric).justification for metric in JUDGE_METRICS}


class ResponseJudge:
    """Scores responses against their source rows using an LLM judge."""

    def __init__(self, model: str = JUDGE_MODELS[0]):
        self.model = model

    def grade(self, row: TrainingRow, parsed: ParsedAnswer) -> ResponseVerdict:
        """Grade one response.

        Params:
            row: The row the response was generated from.
            parsed: The model's answer, already split into its parts.

        Returns:
            The judge's verdict.
        """
        prompt = self._build_prompt(row, parsed)
        return get_llm_client(self.model).generate(prompt, ResponseVerdict)

    def grade_many(
        self,
        graded: list[tuple[TrainingRow, ParsedAnswer]],
        workers: int = JUDGE_WORKERS,
    ) -> list[ResponseVerdict | None]:
        """Grade responses in parallel.

        Params:
            graded: Row and parsed answer pairs to judge.
            workers: Parallel judge calls.

        Returns:
            One verdict per pair in the input order, or None where the judge
            failed, so one bad call cannot lose a whole evaluation run.
        """

        def grade_one(pair: tuple[TrainingRow, ParsedAnswer]) -> ResponseVerdict | None:
            try:
                return self.grade(*pair)
            except Exception as error:
                print(f"  {self.model} judge failed on row {pair[0].id}: {error}")
                return None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(grade_one, graded))

    @staticmethod
    def _build_prompt(row: TrainingRow, parsed: ParsedAnswer) -> str:
        """Format one response and its ground truth into the rubric prompt.

        Params:
            row: The row the response was generated from.
            parsed: The model's answer.

        Returns:
            The prompt text.
        """
        decomposition_text = "\n".join(
            f"  Step {step.id}: {step.instruction} -> {step.answer}" for step in row.decomposition
        )
        return RESPONSE_JUDGE_PROMPT.format(
            instruction=row.instruction,
            search_pool_text=render_labeled_pool(row.search_pool),
            informative_ids=gold_informative_ids(row),
            decomposition_text=decomposition_text,
            reference_response=row.response,
            cited_ids="(none parsed)" if parsed.cited_ids is None else parsed.cited_ids,
            rationale=parsed.rationale or "(none)",
            response=parsed.response or "(none)",
        )
