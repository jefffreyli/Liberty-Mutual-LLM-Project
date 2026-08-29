"""The generated test split, registered as a benchmark so the external datasets and the data the
model was trained on are scored through one code path. It is the only benchmark that keeps its
rows in split order rather than sampling, because those numbers are the reference every transfer
result is read against.
"""

from __future__ import annotations

from pathlib import Path

from src.config import training as train_cfg
from src.evaluation.benchmarks.base import Benchmark
from src.paths import RESULTS_DIR
from src.schema import TrainingRow
from src.training.data import load_rows, split_rows, split_sizes


class Synthetic(Benchmark):
    """Rows held out of every training split by src/training/data.py."""

    name = "synthetic"
    default_limit = None

    @property
    def results_dir(self) -> Path:
        """Keep results at the top level, where every existing file already is.

        Returns:
            The results directory itself, unnested.
        """
        return RESULTS_DIR

    def build_rows(self) -> list[TrainingRow]:
        """Take the test rows that no trainer has seen.

        Returns:
            The canonical test split, in split order.
        """
        rows = load_rows(train_cfg.DATA_PATH)
        test_size, val_size = split_sizes(
            len(rows), train_cfg.TEST_FRACTION, train_cfg.VAL_FRACTION
        )
        _, _, test_rows = split_rows(rows, test_size, val_size, train_cfg.SEED)
        return test_rows

    def build_split(self, seed: int) -> list[TrainingRow]:
        """Take the held out rows without sampling or validating them.

        Keeping split order rather than resampling keeps these scores comparable
        with every result already in artifacts/. Validation is skipped because a
        fifth of these rows are abstention rows with no decomposition and no gold
        chunk by design, which is what `validate_rows` rejects everywhere else.

        Params:
            seed: Unused, since no sampling happens.

        Returns:
            The held out rows, in split order.
        """
        return self.build_rows()
