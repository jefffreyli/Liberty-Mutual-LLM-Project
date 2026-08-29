"""Entry point for evaluation: samples answers for the held out rows from a chosen baseline,
scores them with both the programmatic metrics and the LLM judges, prints the aggregate, and
writes every response and verdict to artifacts/results/.

Run:
    python3 -m scripts.evaluate --run base
    python3 -m scripts.evaluate --run rl --limit 50
    python3 -m scripts.evaluate --run sft --benchmark musique --no-judge
    python3 -m scripts.evaluate --run claude-opus-5 --judge gpt-5.5 --judge claude-opus-5
"""

import argparse
import json

from src.config.models import JUDGE_MODELS, MODELS
from src.evaluation.baselines import BASELINES
from src.evaluation.benchmarks import BENCHMARKS, get_benchmark
from src.evaluation.runner import evaluate, report
from src.training.session import load_api_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=sorted(BASELINES), default="base")
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARKS),
        default="synthetic",
        help="dataset to score on, default the generated test split",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="rows to score, default the benchmark's own"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="override the baseline's generation budget"
    )
    parser.add_argument(
        "--judge",
        action="append",
        choices=sorted(MODELS),
        default=None,
        help=f"judge model, repeatable, default {' '.join(JUDGE_MODELS)}",
    )
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judges")
    parser.add_argument(
        "--rebuild-split",
        action="store_true",
        help="rebuild the benchmark's datasets/<name>/test.jsonl before scoring",
    )
    parser.add_argument("--checkpoint", default=None, help="checkpoint name, default the run's latest")
    args = parser.parse_args()

    judge_models = [] if args.no_judge else (args.judge or list(JUDGE_MODELS))

    load_api_key()
    records = evaluate(
        args.run,
        args.limit,
        args.max_tokens,
        judge_models,
        checkpoint=args.checkpoint,
        benchmark=args.benchmark,
        rebuild_split=args.rebuild_split,
    )
    report(records)

    results_dir = get_benchmark(args.benchmark).results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.run}_{args.checkpoint}" if args.checkpoint else args.run
    output_path = results_dir / f"{name}.json"
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
