"""The one place the repository layout is written down. Every derived file the pipeline writes
lands under artifacts/, split by what produced it, so a single directory holds everything that
can be regenerated and nothing that cannot.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Derived outputs. Gitignored, and safe to delete in full.
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"  # generated data, plus the JSONL derived from it
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"  # Tinker logs, metrics, exported weights
RESULTS_DIR = ARTIFACTS_DIR / "results"  # per row evaluation records

# Materialized evaluation splits, one test.jsonl per benchmark. Written on the
# first run and read by every run after it, so a scored comparison never depends
# on a sample being redrawn the same way twice.
DATASETS_DIR = ROOT / "datasets"
