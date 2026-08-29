"""Pushes a generated run to the HuggingFace Hub as a dataset, flattening the nested
decomposition and search pool into JSON strings so the rows fit a flat schema.

Run:
    python3 -m scripts.upload_to_hf [--repo REPO_ID] [--file RUN_JSON] [--private]
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, Features, Value

from src.config import training as cfg

DEFAULT_REPO = "jefffreyli/multihop-3k"

FEATURES = Features(
    {
        "id": Value("string"),
        "instruction": Value("string"),
        "decomposition": Value("string"),  # JSON-encoded list[dict]
        "search_pool": Value("string"),  # JSON-encoded list[dict]
        "rationale": Value("string"),
        "response": Value("string"),
    }
)


def load_raw_rows(path: Path) -> list[dict]:
    """Read the rows of a run JSON without validating them.

    Params:
        path: Path to the run JSON.

    Returns:
        The raw rows.
    """
    with open(path) as f:
        return json.load(f)["data"]


def build_dataset(rows: list[dict]) -> Dataset:
    """Flatten rows into a Hub-friendly dataset.

    Params:
        rows: Raw rows from a run JSON.

    Returns:
        A dataset whose nested fields are JSON strings.
    """
    flat = [
        {
            "id": row["id"],
            "instruction": row["instruction"],
            "decomposition": json.dumps(row["decomposition"]),
            "search_pool": json.dumps(row["search_pool"]),
            "rationale": row["rationale"],
            "response": row["response"],
        }
        for row in rows
    ]
    return Dataset.from_list(flat, features=FEATURES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hub repo id (username/dataset-name)")
    parser.add_argument("--file", type=Path, default=cfg.DATA_PATH, help="run JSON to upload")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    args = parser.parse_args()

    print(f"Loading {args.file} ...")
    rows = load_raw_rows(args.file)
    print(f"  {len(rows):,} rows loaded")

    dataset = build_dataset(rows)
    print(f"Pushing {len(dataset):,} rows to {args.repo} ...")
    dataset.push_to_hub(args.repo, private=args.private)
    print(f"Done! Dataset available at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
