"""Entry point for judging results that were already sampled: reads the per row records a
previous evaluation wrote, grades their completions with one or more LLM judges, and writes the
verdicts back into the same file. Sampling is the expensive half of an evaluation, so this exists
to add a judge to a finished run without paying for the completions twice.

Run:
    python3 -m scripts.judge_results base sft gpt-4o
    python3 -m scripts.judge_results --judge gpt-5.5 --judge claude-opus-5 sft
"""

import argparse
import json
from pathlib import Path

from src.config.models import JUDGE_MODELS, MODELS
from src.evaluation.judge import ResponseJudge
from src.evaluation.runner import held_out_rows
from src.paths import RESULTS_DIR
from src.training.format import parse_answer
from src.training.session import load_api_key


def judge_file(path: Path, judge_models: list[str]) -> None:
    """Grade one results file in place.

    Params:
        path: Results JSON written by scripts/evaluate.py.
        judge_models: Judges to grade with.

    Raises:
        ValueError: If the file's rows do not match the current test split,
            which means it was produced from a different dataset.
    """
    records = json.loads(path.read_text())
    rows = {row.id: row for row in held_out_rows()}
    if {record["id"] for record in records} != set(rows):
        raise ValueError(
            f"{path.name} was scored on different rows than the current test split; "
            f"re-run scripts.evaluate for this dataset instead of judging stale records"
        )

    ordered = [rows[record["id"]] for record in records]
    parsed = [parse_answer(record["completion"]) for record in records]

    for judge_model in judge_models:
        print(f"  judging {path.name} with {judge_model} ...")
        verdicts = ResponseJudge(judge_model).grade_many(list(zip(ordered, parsed)))
        graded = sum(1 for verdict in verdicts if verdict)
        for record, verdict in zip(records, verdicts):
            record.setdefault("judge", {})[judge_model] = verdict.as_metrics() if verdict else None
            record.setdefault("judge_justifications", {})[judge_model] = (
                verdict.justifications() if verdict else None
            )
        print(f"    {graded}/{len(records)} rows graded")

    path.write_text(json.dumps(records, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge already sampled evaluation results.")
    parser.add_argument("runs", nargs="+", help="results file stems, such as base or gpt-4o")
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
    for run in args.runs:
        judge_file(RESULTS_DIR / f"{run}.json", judge_models)


if __name__ == "__main__":
    main()
