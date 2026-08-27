"""Entry point for evaluation: samples answers for the held out rows from a chosen set of
weights, scores them with both the programmatic metrics and the LLM judge, prints the aggregate,
and writes every response and verdict to artifacts/results/.

Run:
    python3 -m scripts.evaluate --run base
    python3 -m scripts.evaluate --run rl --limit 50
    python3 -m scripts.evaluate --run sft --no-judge
"""

import argparse
import json

from src.config import training as cfg
from src.evaluation.runner import evaluate, report
from src.paths import RESULTS_DIR
from src.training.session import load_api_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["base", *sorted(cfg.RUN_LOG_PATHS)], default="base")
    parser.add_argument("--limit", type=int, default=None, help="rows to score, default all")
    parser.add_argument("--max-tokens", type=int, default=cfg.RL_MAX_TOKENS)
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    args = parser.parse_args()

    load_api_key()
    records = evaluate(args.run, args.limit, args.max_tokens, use_judge=not args.no_judge)
    report(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{args.run}.json"
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nWrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
