"""Entry point for combining generated runs: concatenates the rows of several run JSONs into one,
reassigns row ids so they stay unique, and sums the token cost of the runs that produced them.

Run:
    python3 -m scripts.merge_runs --out artifacts/runs/combined_3750.json run_a.json run_b.json
"""

import argparse
import json
from pathlib import Path

from src.generation.generator import ROW_ID_PREFIX


def parse_args() -> argparse.Namespace:
    """Read the runs to merge and where to write them.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Merge generated run JSONs into one run.")
    parser.add_argument("runs", nargs="+", type=Path, help="Run JSONs to merge, in order.")
    parser.add_argument("--out", type=Path, required=True, help="Destination JSON path.")
    return parser.parse_args()


def merge(run_paths: list[Path]) -> dict:
    """Concatenate the rows of several runs into one payload.

    Params:
        run_paths: Run JSONs to merge, in order.

    Returns:
        A run payload whose config records the sources and the summed cost.

    Raises:
        ValueError: If any file is not a run JSON.
    """
    rows: list[dict] = []
    sources: list[dict] = []
    input_cost, output_cost = 0.0, 0.0

    for path in run_paths:
        with open(path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or "data" not in payload:
            raise ValueError(f"Expected a dict with a 'data' key at {path}")
        run_config = payload.get("config", {})
        input_cost += run_config.get("input_cost", 0.0)
        output_cost += run_config.get("output_cost", 0.0)
        sources.append({"file": path.name, "num_rows": len(payload["data"])})
        rows.extend(payload["data"])

    # Row ids are only unique within a run, so they are reassigned across the merge.
    for i, row in enumerate(rows):
        row["id"] = f"{ROW_ID_PREFIX}{i:05d}"

    unanswerable = sum(
        1 for row in rows if not any(chunk["is_informative"] for chunk in row["search_pool"])
    )
    return {
        "config": {
            "num_rows": len(rows),
            "sources": sources,
            "unanswerable_rows": unanswerable,
            "unanswerable_fraction": unanswerable / len(rows) if rows else 0.0,
            "input_cost": input_cost,
            "output_cost": output_cost,
        },
        "data": rows,
    }


def main() -> None:
    args = parse_args()
    payload = merge(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    config = payload["config"]
    print(f"Merged {len(args.runs)} runs into {args.out}")
    print(f"  rows: {config['num_rows']}")
    print(
        f"  unanswerable: {config['unanswerable_rows']} "
        f"({config['unanswerable_fraction']:.1%})"
    )
    for source in config["sources"]:
        print(f"  {source['file']}: {source['num_rows']} rows")


if __name__ == "__main__":
    main()
