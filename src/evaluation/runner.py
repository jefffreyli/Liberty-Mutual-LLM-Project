"""Scores one set of weights on the held out rows: samples an answer per row, grades it with
both the programmatic metrics and the LLM judge, and assembles the per row records that
scripts/evaluate.py prints and writes.
"""

from statistics import mean

from src.config import training as train_cfg
from src.config.models import JUDGE_MODEL
from src.evaluation.judge import ResponseJudge, ResponseVerdict
from src.training.metrics import Grade, grade_answer
from src.evaluation.sampler import ResponseSampler, resolve_model_path
from src.llm.client import get_aggregate_cost
from src.schema import TrainingRow
from src.training.data import load_rows, split_rows, split_sizes
from src.training.format import ParsedAnswer, parse_answer

REWARD_WEIGHTS = {
    "citation_weight": train_cfg.REWARD_CITATION_WEIGHT,
    "answer_weight": train_cfg.REWARD_ANSWER_WEIGHT,
    "leakage_weight": train_cfg.REWARD_LEAKAGE_WEIGHT,
}


def held_out_rows(limit: int | None = None) -> list[TrainingRow]:
    """Take the test rows that no trainer has seen.

    Params:
        limit: Maximum number of rows to score, or None for the whole test split.

    Returns:
        Rows from the canonical test split.
    """
    rows = load_rows(train_cfg.DATA_PATH)
    test_size, val_size = split_sizes(
        len(rows), train_cfg.TEST_FRACTION, train_cfg.VAL_FRACTION
    )
    _, _, test_rows = split_rows(rows, test_size, val_size, train_cfg.SEED)
    return test_rows[:limit] if limit else test_rows


def build_records(
    rows: list[TrainingRow],
    completions: list[str],
    grades: list[Grade],
    verdicts: list[ResponseVerdict | None],
) -> list[dict]:
    """Assemble the per row record written to disk.

    Params:
        rows: The evaluated rows.
        completions: Raw model answers.
        grades: Programmatic grades.
        verdicts: Judge verdicts, with None where the judge failed.

    Returns:
        One record per row.
    """
    return [
        {
            "id": row.id,
            "instruction": row.instruction,
            "completion": completion,
            "metrics": grade.as_metrics(),
            "judge": verdict.as_metrics() if verdict else None,
            "judge_justifications": verdict.justifications() if verdict else None,
        }
        for row, completion, grade, verdict in zip(rows, completions, grades, verdicts)
    ]


def report(records: list[dict]) -> None:
    """Print the mean of every metric that was collected.

    Params:
        records: Per row records.
    """
    print(f"\nScored {len(records)} rows")
    print("  Programmatic:")
    for key in records[0]["metrics"]:
        print(f"    {key}: {mean(r['metrics'][key] for r in records):.3f}")

    judged = [r["judge"] for r in records if r["judge"]]
    if not judged:
        return
    print(f"  Judge ({len(judged)}/{len(records)} graded):")
    for key in judged[0]:
        print(f"    {key}: {mean(j[key] for j in judged):.3f}")


def evaluate(
    run: str,
    limit: int | None,
    max_tokens: int,
    use_judge: bool,
    checkpoint: str | None = None,
) -> list[dict]:
    """Sample and score one set of weights.

    Params:
        run: "base", or a key of train_cfg.RUN_LOG_PATHS.
        limit: Maximum rows to score, or None for the whole test split.
        max_tokens: Generation budget per answer.
        use_judge: Whether to run the LLM judge on top of the programmatic metrics.
        checkpoint: Name of one checkpoint to score, or None for the run's latest.

    Returns:
        Per row records.
    """
    rows = held_out_rows(limit)
    label = f"{run}/{checkpoint}" if checkpoint else run
    print(f"Evaluating {label} on {len(rows)} held out rows")

    model_path = resolve_model_path(run, checkpoint)
    completions = ResponseSampler(model_path, max_tokens=max_tokens).sample(rows)
    parsed: list[ParsedAnswer] = [parse_answer(completion) for completion in completions]
    grades = [
        grade_answer(answer, row, **REWARD_WEIGHTS) for row, answer in zip(rows, parsed)
    ]

    verdicts: list[ResponseVerdict | None] = [None] * len(rows)
    if use_judge:
        print(f"Judging with {JUDGE_MODEL} ...")
        verdicts = ResponseJudge().grade_many(list(zip(rows, parsed)))
        input_cost, output_cost = get_aggregate_cost()
        print(f"Judge cost (input/output): ${input_cost:.4f} / ${output_cost:.4f}")

    return build_records(rows, completions, grades, verdicts)
