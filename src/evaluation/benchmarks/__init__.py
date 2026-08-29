"""The set of datasets an evaluation run can score, keyed by the name `--benchmark` takes. Adding
one means writing a Benchmark subclass and listing it here, the same shape as adding a Baseline.
"""

from src.evaluation.benchmarks.base import Benchmark
from src.evaluation.benchmarks.browsecomp import BrowseCompPlus
from src.evaluation.benchmarks.finqa import FinQA
from src.evaluation.benchmarks.hotpotqa import HotpotQA
from src.evaluation.benchmarks.longbench import LongBench
from src.evaluation.benchmarks.musique import MuSiQue
from src.evaluation.benchmarks.synthetic import Synthetic

_BENCHMARKS = (Synthetic(), MuSiQue(), HotpotQA(), FinQA(), LongBench(), BrowseCompPlus())

BENCHMARKS: dict[str, Benchmark] = {benchmark.name: benchmark for benchmark in _BENCHMARKS}


def get_benchmark(name: str) -> Benchmark:
    """Look up a benchmark by the name the command line takes.

    Params:
        name: A key of BENCHMARKS.

    Returns:
        The benchmark.

    Raises:
        KeyError: If no benchmark goes by that name.
    """
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{name}'. Known: {', '.join(sorted(BENCHMARKS))}")
    return BENCHMARKS[name]


__all__ = ["BENCHMARKS", "Benchmark", "get_benchmark"]
