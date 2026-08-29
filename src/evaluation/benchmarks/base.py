"""Adapts an external benchmark into the TrainingRow shape the rest of the evaluation stack
already reads, so every baseline answers identical rows and is graded by the same metrics.
A subclass supplies only the field mapping for its dataset in `build_rows`; the base class assigns
pool IDs, draws a stratified sample, and validates the result before any model sees it.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from pathlib import Path

from src.config import training as train_cfg
from src.evaluation.judge import GroundingVerdict, ResponseVerdict
from src.paths import DATASETS_DIR, RESULTS_DIR
from src.schema import SearchResult, TrainingRow


class Benchmark(ABC):
    """One dataset an evaluation run can score.

    Class attributes:
        name: The key `scripts/evaluate.py --benchmark` takes, and the
            subdirectory its results are written to.
        verdict_model: The judge verdict shape this dataset can support.
            `hop_completeness` needs a real per hop decomposition, so only a
            dataset that ships one may name the five metric shape.
        reference_label: How the judge prompt describes the reference answer,
            which is a written response on some datasets and a bare number or a
            paragraph id on others.
        default_limit: Rows to sample when the command line does not say.
        few_shot_override: Exemplars to prompt with on this benchmark, or None
            to use whatever the baseline asks for.
    """

    name: str
    verdict_model: type[GroundingVerdict] = ResponseVerdict
    reference_label: str = "written from the informative paragraphs only"
    default_limit: int | None = 300
    few_shot_override: int | None = None

    @property
    def results_dir(self) -> Path:
        """Where this benchmark's per row records are written.

        Returns:
            The directory, which may not exist yet.
        """
        return RESULTS_DIR / self.name

    @abstractmethod
    def build_rows(self) -> list[TrainingRow]:
        """Fetch the dataset and map every usable example onto a row.

        Returns:
            Every row the benchmark can offer, before sampling.
        """

    @property
    def split_path(self) -> Path:
        """Where this benchmark's materialized evaluation split lives.

        Returns:
            The test.jsonl path, which may not exist yet.
        """
        return DATASETS_DIR / self.name / "test.jsonl"

    def load(
        self, limit: int | None = None, seed: int = train_cfg.SEED, rebuild: bool = False
    ) -> list[TrainingRow]:
        """Read the materialized split, building it on first use.

        The split is always built at the benchmark's full size and `limit` takes
        a prefix of it, so a smoke test run with a small limit cannot leave a
        truncated split behind for the next run to score against.

        Params:
            limit: Rows to return, or None for the whole split.
            seed: Sampling seed, used only when the split is being built. It
                defaults to the one seed the trainers and evaluation share, so a
                rebuild cannot quietly produce a different split.
            rebuild: Rebuild and overwrite the split even if it already exists.

        Returns:
            The rows this benchmark scores.
        """
        if rebuild or not self.split_path.exists():
            rows = self.build_split(seed)
            self.split_path.parent.mkdir(parents=True, exist_ok=True)
            self.split_path.write_text(
                "".join(f"{row.model_dump_json()}\n" for row in rows), encoding="utf-8"
            )
        else:
            with open(self.split_path, encoding="utf-8") as f:
                rows = [TrainingRow.model_validate_json(line) for line in f if line.strip()]
        return rows[:limit] if limit else rows

    def build_split(self, seed: int) -> list[TrainingRow]:
        """Adapt, filter, sample, and validate the rows to materialize.

        Params:
            seed: Sampling seed.

        Returns:
            The validated sample, in a shuffled but reproducible order.
        """
        rows = [row for row in self.build_rows() if is_discriminative(row)]
        rows = stratified_sample(rows, self.default_limit, seed, key=lambda row: row.source)
        validate_rows(rows, self.name)
        return rows


def build_pool(chunks: Sequence[tuple[str, str, bool]]) -> list[SearchResult]:
    """Turn (title, text, is_informative) triples into a pool with sequential IDs.

    Source order is preserved rather than shuffled: no adapted dataset orders its
    gold paragraphs first, and LongBench's paragraph numbering is the answer.

    Params:
        chunks: One triple per search result, in the order the model will see them.

    Returns:
        The search pool, with IDs 0..n-1.
    """
    return [
        SearchResult(id=index, title=title, text=text, is_informative=informative)
        for index, (title, text, informative) in enumerate(chunks)
    ]


def stratified_sample(
    rows: list[TrainingRow], limit: int | None, seed: int, key
) -> list[TrainingRow]:
    """Draw a sample that holds each stratum's share of the whole.

    Params:
        rows: Every row the benchmark built.
        limit: Rows to keep, or None to keep all of them.
        seed: Shuffle and selection seed.
        key: Maps a row to its stratum.

    Returns:
        The sample, shuffled. Every row when `limit` is None or covers the set.
    """
    if limit is None or limit >= len(rows):
        return list(rows)

    strata: dict[Hashable, list[TrainingRow]] = {}
    for row in rows:
        strata.setdefault(key(row), []).append(row)

    # Largest remainder, so the sample keeps each stratum's proportion and still
    # sums to exactly `limit`.
    exact = {name: len(group) * limit / len(rows) for name, group in strata.items()}
    counts = {name: int(share) for name, share in exact.items()}
    slack = limit - sum(counts.values())
    for name in sorted(strata, key=lambda name: exact[name] - counts[name], reverse=True)[:slack]:
        counts[name] += 1

    rng = random.Random(seed)
    sample = [row for name, group in strata.items() for row in rng.sample(group, counts[name])]
    rng.shuffle(sample)
    return sample


def is_discriminative(row: TrainingRow) -> bool:
    """Whether a row actually poses a discrimination problem.

    A handful of HotpotQA validation rows ship their two gold paragraphs with no
    distractors at all, where citing the whole pool scores a perfect F1 without
    deciding anything.

    Params:
        row: A candidate row.

    Returns:
        True when the pool holds at least one gold chunk and one distractor.
    """
    return {chunk.is_informative for chunk in row.search_pool} == {True, False}


def validate_rows(rows: list[TrainingRow], name: str) -> None:
    """Reject rows that would score well for the wrong reason.

    `answer_coverage` returns 1.0 for a row with no decomposition, which is
    correct for a synthetic abstention row and would otherwise hand every model
    half the reward for free. A pool with no distractor hands out citation F1 the
    same way. Cited IDs are the integers the model reads off the rendered pool,
    so the IDs have to be the pool's own indices.

    Params:
        rows: The sampled rows.
        name: Benchmark name, for the error message.

    Raises:
        ValueError: If a row has no decomposition, a pool that is all gold or
            all distractor, or a pool whose IDs are not 0..n-1.
    """
    for row in rows:
        if not row.decomposition:
            raise ValueError(f"{name} row {row.id} has no decomposition to score coverage against")
        if not is_discriminative(row):
            raise ValueError(
                f"{name} row {row.id} has no informative chunk or no distractor to reject"
            )
        ids = [chunk.id for chunk in row.search_pool]
        if ids != list(range(len(ids))):
            raise ValueError(f"{name} row {row.id} has non sequential pool IDs: {ids}")
