"""Entry point for data generation: builds the configured number of rows with DatasetGenerator
and writes them, along with the run's settings and cost, to a timestamped JSON file under
artifacts/runs/.

Run:
    python3 -m scripts.generate
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import generation as config
from src.config.models import DEFAULT_MODEL, TOKEN_PRICE
from src.generation.generator import DatasetGenerator
from src.paths import RUNS_DIR


def write_run(generator: DatasetGenerator, output_path: Path) -> None:
    """Write the generated rows and the settings that produced them.

    Params:
        generator: A generator whose rows have already been produced.
        output_path: Destination JSON path.
    """
    input_cost, output_cost = generator.get_total_cost()
    payload = {
        "config": {
            "num_rows": generator.num_rows,
            "seed_dataset_name": config.SEED_DATASET_NAME or None,
            "unanswerable_fraction": generator.unanswerable_fraction,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "default_model": DEFAULT_MODEL,
            "token_price": TOKEN_PRICE,
        },
        "data": [row.model_dump() for row in generator.rows],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(generator.rows)} rows to {output_path}")
    print(f"Token cost (input/output): ${input_cost:.4f} / ${output_cost:.4f}")


def parse_args() -> argparse.Namespace:
    """Read the run settings that differ between runs.

    Returns:
        The parsed arguments, defaulting to src/config/generation.py.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic training rows.")
    parser.add_argument("--num-rows", type=int, default=config.NUM_ROWS)
    parser.add_argument(
        "--unanswerable-fraction",
        type=float,
        default=config.UNANSWERABLE_FRACTION,
        help="Share of rows whose search pool holds no informative chunk.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Destination JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.out or RUNS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print(f"Generating {args.num_rows} training rows...")
    generator = DatasetGenerator(
        num_rows=args.num_rows,
        seed_dataset_name=config.SEED_DATASET_NAME,
        unanswerable_fraction=args.unanswerable_fraction,
    )
    generator.generate_dataset()
    write_run(generator, output_path)


if __name__ == "__main__":
    main()
