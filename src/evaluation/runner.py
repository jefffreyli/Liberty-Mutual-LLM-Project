"""Scores one baseline on the held out rows: samples an answer per row, grades it with the
programmatic metrics and every requested LLM judge, and assembles the per row records that
scripts/evaluate.py prints and writes.
"""

from dataclasses import replace
from statistics import mean

from src.config import training as train_cfg
from src.evaluation.baselines import get_baseline
from src.evaluation.judge import ResponseJudge, ResponseVerdict
from src.evaluation.sampler import build_sampler
from src.llm.client import get_aggregate_cost
from src.schema import TrainingRow
from src.training.data import load_rows, split_rows, split_sizes
from src.training.format import ParsedAnswer, parse_answer
from src.training.metrics import Grade, grade_answer

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
    verdicts: dict[str, list[ResponseVerdict | None]],
) -> list[dict]:
    """Assemble the per row record written to disk.

    Params:
        rows: The evaluated rows.
        completions: Raw model answers.
        grades: Programmatic grades.
        verdicts: Judge model to its verdicts, with None where that judge failed.

    Returns:
        One record per row.
    """
    records = []
    for index, (row, completion, grade) in enumerate(zip(rows, completions, grades)):
        judged = {
            model: model_verdicts[index]
            for model, model_verdicts in verdicts.items()
            if model_verdicts[index] is not None
        }
        records.append(
            {
                "id": row.id,
                # Carried so the report can split the two row types, which are
                # different tasks: one is grounded answering, the other is
                # recognizing that the pool cannot support an answer at all.
                "answerable": row.is_answerable,
                "instruction": row.instruction,
                "completion": completion,
                "metrics": grade.as_metrics(),
                "judges": {model: verdict.as_metrics() for model, verdict in judged.items()},
                "judge_justifications": {
                    model: verdict.justifications() for model, verdict in judged.items()
                },
            }
        )
    return records


def _print_means(records: list[dict], label: str) -> None:
    """Print the mean of every metric collected for a set of records.

    Params:
        records: Per row records, assumed non-empty.
        label: Heading naming the subset.
    """
    print(f"  {label} ({len(records)} rows):")
    print("    Programmatic:")
    for key in records[0]["metrics"]:
        print(f"      {key}: {mean(r['metrics'][key] for r in records):.3f}")

    for model in sorted({model for r in records for model in r["judges"]}):
        judged = [r["judges"][model] for r in records if model in r["judges"]]
        print(f"    Judge {model} ({len(judged)}/{len(records)} graded):")
        for key in judged[0]:
            print(f"      {key}: {mean(j[key] for j in judged):.3f}")


def report(records: list[dict]) -> None:
    """Print the aggregate overall and split by row type.

    Params:
        records: Per row records.
    """
    print(f"\nScored {len(records)} rows")
    _print_means(records, "All")
    for label, subset in (
        ("Answerable", [r for r in records if r["answerable"]]),
        ("Unanswerable", [r for r in records if not r["answerable"]]),
    ):
        if subset:
            _print_means(subset, label)


def evaluate(
    run: str,
    limit: int | None,
    max_tokens: int | None,
    judge_models: list[str],
    checkpoint: str | None = None,
) -> list[dict]:
    """Sample and score one baseline.

    Params:
        run: A key of src.evaluation.baselines.BASELINES.
        limit: Maximum rows to score, or None for the whole test split.
        max_tokens: Generation budget per answer, or None for the baseline's own.
        judge_models: Judges to grade with, empty to skip judging entirely.
        checkpoint: Name of one checkpoint to score, or None for the run's
            latest. Ignored by API baselines.

    Returns:
        Per row records.
    """
    rows = held_out_rows(limit)
    baseline = get_baseline(run)
    if max_tokens is not None:
        baseline = replace(baseline, max_tokens=max_tokens)

    label = f"{run}/{checkpoint}" if checkpoint else run
    print(
        f"Evaluating {label} on {len(rows)} held out rows "
        f"({baseline.kind}, {baseline.few_shot}-shot)"
    )
    completions = build_sampler(baseline, checkpoint).sample(rows)
    parsed: list[ParsedAnswer] = [parse_answer(completion) for completion in completions]
    grades = [
        grade_answer(answer, row, **REWARD_WEIGHTS) for row, answer in zip(rows, parsed)
    ]

    verdicts: dict[str, list[ResponseVerdict | None]] = {}
    for judge_model in judge_models:
        print(f"Judging with {judge_model} ...")
        verdicts[judge_model] = ResponseJudge(judge_model).grade_many(list(zip(rows, parsed)))
    if judge_models:
        input_cost, output_cost = get_aggregate_cost()
        print(f"API cost (input/output): ${input_cost:.4f} / ${output_cost:.4f}")

    return build_records(rows, completions, grades, verdicts)
