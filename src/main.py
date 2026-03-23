"""CLI entry point for SAIL synthetic data generation."""

import json
from datetime import datetime
from pathlib import Path

from .data.generator import generate_dataset

RUNS_DIR = Path(__file__).parent.parent / "runs"


def main():
    # output path
    RUNS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(RUNS_DIR / f"run_{timestamp}.json")

    # config
    num_rows = 10

    # generate dataset
    print(f"Generating {num_rows} training rows...")
    rows = generate_dataset(num_rows)

    # write to JSON
    data = [row.model_dump() for row in rows]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
