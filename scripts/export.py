"""Entry point for weight export: downloads a finished run's LoRA adapter, merges it into the
base model in HuggingFace format, and optionally pushes the result to the Hub.

Run:
    python3 -m scripts.export --run rl
    python3 -m scripts.export --run sft --push
"""

import argparse

from src.config import training as cfg
from src.training.export import export
from src.training.session import load_api_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=sorted(cfg.RUN_LOG_PATHS), default="rl")
    parser.add_argument("--push", action="store_true", help="push the merged model to the Hub")
    parser.add_argument("--repo", default=cfg.HF_REPO_ID)
    parser.add_argument("--public", action="store_true", help="make the pushed repo public")
    args = parser.parse_args()

    load_api_key()
    export(args.run, args.repo if args.push else None, private=not args.public)


if __name__ == "__main__":
    main()
