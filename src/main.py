"""CLI entry point for SAIL synthetic data generation."""

import json
from datetime import datetime
from pathlib import Path

from src import config
from src.data.dataset import Dataset

RUNS_DIR = Path(__file__).parent.parent / "runs"


def main():
    # output path
    RUNS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(RUNS_DIR / f"run_{timestamp}.json")

    # config
    num_rows = config.NUM_ROWS

    # generate dataset
    print(f"Generating {num_rows} training rows...")
    dataset = Dataset(num_rows=num_rows)
    dataset.generate_dataset()
    input_cost, output_cost = dataset.get_total_cost()

    # write to JSON
    data = [row.model_dump() for row in dataset.rows]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(dataset.rows)} rows to {output_path}")
    print(f"Token cost (input/output): ${input_cost:.4f} / ${output_cost:.4f}")


if __name__ == "__main__":
    main()
