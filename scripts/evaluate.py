"""Entry point for evaluation: samples answers for the held out rows from a chosen baseline,
scores them with both the programmatic metrics and the LLM judges, prints the aggregate, and
writes every response and verdict to artifacts/results/.

Run:
    python3 -m scripts.evaluate --run base
    python3 -m scripts.evaluate --run rl --limit 50
    python3 -m scripts.evaluate --run sft --no-judge
    python3 -m scripts.evaluate --run claude-opus-5 --judge gpt-5.5 --judge claude-opus-5
"""

import argparse
import json

from src.config.models import JUDGE_MODELS, MODELS
from src.evaluation.baselines import BASELINES
from src.evaluation.runner import evaluate, report
from src.paths import RESULTS_DIR
from src.training.session import load_api_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=sorted(BASELINES), default="base")
    parser.add_argument("--limit", type=int, default=None, help="rows to score, default all")
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
    parser.add_argument("--checkpoint", default=None, help="checkpoint name, default the run's latest")
    args = parser.parse_args()

    judge_models = [] if args.no_judge else (args.judge or list(JUDGE_MODELS))

    load_api_key()
    records = evaluate(
        args.run, args.limit, args.max_tokens, judge_models, checkpoint=args.checkpoint
    )
    report(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{args.run}_{args.checkpoint}" if args.checkpoint else args.run
    output_path = RESULTS_DIR / f"{name}.json"
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
