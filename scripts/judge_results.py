"""Entry point for judging results that were already sampled: reads the per row records a
previous evaluation wrote, grades their completions with one or more LLM judges, and writes the
verdicts back into the same file. Sampling is the expensive half of an evaluation, so this exists
to add a judge to a finished run without paying for the completions twice.

Run:
    python3 -m scripts.judge_results base sft gpt-4o
    python3 -m scripts.judge_results --benchmark musique --judge claude-opus-5 sft rl
"""

import argparse
import json
from pathlib import Path

from src.config import training as train_cfg
from src.config.models import JUDGE_MODELS, MODELS
from src.evaluation.benchmarks import BENCHMARKS, Benchmark, get_benchmark
from src.evaluation.judge import ResponseJudge
from src.paths import RESULTS_DIR
from src.training.format import parse_answer
from src.training.session import load_api_key


def judge_file(path: Path, benchmark: Benchmark, judge_models: list[str]) -> None:
    """Grade one results file in place.

    Params:
        path: Results JSON written by scripts/evaluate.py.
        benchmark: The benchmark it was scored on, which names the rows, the
            rubric shape, and how the reference answer is described.
        judge_models: Judges to grade with.

    Raises:
        ValueError: If the file's rows do not match the benchmark's current
            rows, which means it was produced from a different dataset.
    """
    records = json.loads(path.read_text())
    rows = {row.id: row for row in benchmark.load(seed=train_cfg.SEED)}
    if {record["id"] for record in records} != set(rows):
        raise ValueError(
            f"{path.name} was scored on different rows than {benchmark.name} now loads; "
            f"re-run scripts.evaluate for this benchmark instead of judging stale records"
        )

    ordered = [rows[record["id"]] for record in records]
    parsed = [parse_answer(record["completion"]) for record in records]

    for judge_model in judge_models:
        print(f"  judging {path.name} with {judge_model} ...")
        judge = ResponseJudge(judge_model, benchmark.verdict_model, benchmark.reference_label)
        verdicts = judge.grade_many(list(zip(ordered, parsed)))
        graded = sum(1 for verdict in verdicts if verdict)
        for record, verdict in zip(records, verdicts):
            record.setdefault("judges", {})[judge_model] = verdict.as_metrics() if verdict else None
            record.setdefault("judge_justifications", {})[judge_model] = (
                verdict.justifications() if verdict else None
            )
        print(f"    {graded}/{len(records)} rows graded")

    path.write_text(json.dumps(records, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge already sampled evaluation results.")
    parser.add_argument("runs", nargs="+", help="results file stems, such as base or gpt-4o")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="synthetic")
    parser.add_argument(
        "--judge",
        action="append",
        choices=sorted(MODELS),
        default=None,
        help=f"judge model, repeatable, default {' '.join(JUDGE_MODELS)}",
    )
    args = parser.parse_args()
    judge_models = args.judge or list(JUDGE_MODELS)

    load_api_key()
    benchmark = get_benchmark(args.benchmark)
    results_dir = RESULTS_DIR / benchmark.name if benchmark.name != "synthetic" else RESULTS_DIR
    for run in args.runs:
        judge_file(results_dir / f"{run}.json", benchmark, judge_models)


if __name__ == "__main__":
    main()
