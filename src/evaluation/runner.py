"""Scores one baseline on one benchmark: loads the rows, samples an answer per row, grades it with
the programmatic metrics and every requested LLM judge, and assembles the per row records that
scripts/evaluate.py prints and writes. The benchmark decides which rows are scored and which judge
rubric applies, so the generated test split and the external datasets run through one path.
"""

from dataclasses import replace
from statistics import mean

from src.config import training as train_cfg
from src.config.models import MIN_JUDGE_COVERAGE
from src.evaluation.baselines import get_baseline
from src.evaluation.benchmarks import Benchmark, get_benchmark
from src.evaluation.judge import GroundingVerdict, ResponseJudge
from src.evaluation.sampler import build_sampler
from src.llm.client import get_aggregate_cost
from src.schema import TrainingRow
from src.training.format import ParsedAnswer, parse_answer
from src.training.metrics import Grade, grade_answer

REWARD_WEIGHTS = {
    "citation_weight": train_cfg.REWARD_CITATION_WEIGHT,
    "answer_weight": train_cfg.REWARD_ANSWER_WEIGHT,
    "leakage_weight": train_cfg.REWARD_LEAKAGE_WEIGHT,
}

# Reported per row but not in the aggregate: a malformed answer already scores
# zero through grade_answer, so the mean adds nothing to the comparison.
HIDDEN_METRICS = frozenset({"format_ok"})


def build_records(
    rows: list[TrainingRow],
    completions: list[str],
    grades: list[Grade],
    verdicts: dict[str, list[GroundingVerdict | None]],
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
                "source": row.source,
                # Carried so the report can split the two row types, which are
                # different tasks: one is grounded answering, the other is
                # recognizing that the pool cannot support an answer at all.
                "answerable": row.is_answerable,
                # A parsed but empty ID list. On an answerable row this is always
                # wrong, which is what makes it worth reporting separately.
                "abstained": parse_answer(completion).cited_ids == [],
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
        if key not in HIDDEN_METRICS:
            print(f"      {key}: {mean(r['metrics'][key] for r in records):.3f}")

    answerable = [r for r in records if r["answerable"]]
    if answerable:
        rate = mean(float(r["abstained"]) for r in answerable)
        print(f"      false_abstention: {rate:.3f}")

    for model in sorted({model for r in records for model in r["judges"]}):
        judged = [r["judges"][model] for r in records if model in r["judges"]]
        print(f"    Judge {model} ({len(judged)}/{len(records)} graded):")
        for key in judged[0]:
            print(f"      {key}: {mean(j[key] for j in judged):.3f}")


def report(records: list[dict]) -> None:
    """Print the aggregate overall and, where the benchmark has both, by row type.

    Params:
        records: Per row records.
    """
    print(f"\nScored {len(records)} rows")
    _print_means(records, "All")

    # Every external benchmark is answerable throughout, where the split would
    # only restate the overall aggregate.
    if len({r["answerable"] for r in records}) < 2:
        return
    for label, subset in (
        ("Answerable", [r for r in records if r["answerable"]]),
        ("Unanswerable", [r for r in records if not r["answerable"]]),
    ):
        if subset:
            _print_means(subset, label)


def _judge_all(
    benchmark: Benchmark,
    judge_models: list[str],
    graded: list[tuple[TrainingRow, ParsedAnswer]],
) -> dict[str, list[GroundingVerdict | None]]:
    """Grade every response with each requested judge.

    Params:
        benchmark: The benchmark being scored, which names the rubric shape and
            how the reference answer should be described.
        judge_models: Judges to grade with.
        graded: Row and parsed answer pairs.

    Returns:
        Judge model to its verdicts, with None where that judge failed.

    Raises:
        RuntimeError: If a judge graded less than MIN_JUDGE_COVERAGE of the rows,
            where the aggregate would silently describe only the rows that worked.
    """
    verdicts: dict[str, list[GroundingVerdict | None]] = {}
    for judge_model in judge_models:
        print(f"Judging with {judge_model} ...")
        judge = ResponseJudge(judge_model, benchmark.verdict_model, benchmark.reference_label)
        results = judge.grade_many(graded)
        coverage = sum(verdict is not None for verdict in results) / len(results)
        if coverage < MIN_JUDGE_COVERAGE:
            raise RuntimeError(
                f"{judge_model} graded only {coverage:.0%} of rows, "
                f"below the {MIN_JUDGE_COVERAGE:.0%} floor"
            )
        verdicts[judge_model] = results
    return verdicts


def evaluate(
    run: str,
    limit: int | None,
    max_tokens: int | None,
    judge_models: list[str],
    checkpoint: str | None = None,
    benchmark: str = "synthetic",
    rebuild_split: bool = False,
) -> list[dict]:
    """Sample and score one baseline on one benchmark.

    Params:
        run: A key of src.evaluation.baselines.BASELINES.
        limit: Rows to score, or None for the benchmark's own default.
        max_tokens: Generation budget per answer, or None for the baseline's own.
        judge_models: Judges to grade with, empty to skip judging entirely.
        checkpoint: Name of one checkpoint to score, or None for the run's
            latest. Ignored by API baselines.
        benchmark: A key of src.evaluation.benchmarks.BENCHMARKS.
        rebuild_split: Rebuild the benchmark's materialized split first.

    Returns:
        Per row records.
    """
    bench = get_benchmark(benchmark)
    rows = bench.load(limit, train_cfg.SEED, rebuild=rebuild_split)

    baseline = get_baseline(run)
    if max_tokens is not None:
        baseline = replace(baseline, max_tokens=max_tokens)
    # A long pool leaves no room for the usual exemplars, so the benchmark can
    # cap them. Every baseline on that benchmark is capped the same way.
    if bench.few_shot_override is not None and baseline.few_shot:
        baseline = replace(baseline, few_shot=bench.few_shot_override)

    label = f"{run}/{checkpoint}" if checkpoint else run
    print(
        f"Evaluating {label} on {len(rows)} {bench.name} rows "
        f"({baseline.kind}, {baseline.few_shot}-shot)"
    )
    completions = build_sampler(baseline, checkpoint).sample(rows)
    parsed: list[ParsedAnswer] = [parse_answer(completion) for completion in completions]
    grades = [grade_answer(answer, row, **REWARD_WEIGHTS) for row, answer in zip(rows, parsed)]

    verdicts = _judge_all(bench, judge_models, list(zip(rows, parsed)))
    if judge_models:
        input_cost, output_cost = get_aggregate_cost()
        print(f"API cost (input/output): ${input_cost:.4f} / ${output_cost:.4f}")

    return build_records(rows, completions, grades, verdicts)
