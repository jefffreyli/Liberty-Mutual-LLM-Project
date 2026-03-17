"""CLI entry point for SAIL synthetic data generation."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from data_curation.generator import generate_dataset

RUNS_DIR = Path(__file__).parent / "runs"

def main():
    """Parse args, generate dataset, write to JSON (default: runs/run_<timestamp>.json)."""
    parser = argparse.ArgumentParser(description="Generate SAIL training data")
    args = parser.parse_args()

    # output path
    RUNS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(RUNS_DIR / f"run_{timestamp}.json")

    print(f"Generating {args.num_rows} SAIL training rows...")
    rows = generate_dataset(args.num_rows)

    data = [row.model_dump() for row in rows]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
