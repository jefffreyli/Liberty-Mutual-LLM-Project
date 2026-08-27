"""Generates training rows by prompting the teacher model for an instruction, its informative
paragraphs, and adversarial distractors, then keeps only the rows that pass the rubric gate,
running many rows concurrently across a thread pool. A configured share of the rows are built
unanswerable, with a pool of distractors and no informative chunk at all.
"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import generation as config
from src.config.generation import MAX_RETRIES_PER_ROW
from src.generation.rubric import evaluate_row
from src.generation.instruction import (
    build_decomposition,
    build_informative_chunks,
    generate_grounded_response,
    generate_instruction_bundle,
    generate_rationale,
)
from src.generation.noise import build_search_pool, generate_distractors
from src.generation.seed_loader import SeedLoader
from src.generation.unanswerable import build_unanswerable_row
from src.schema import TrainingRow
from src.schema.seed import SeedExample
from src.llm.client import get_aggregate_cost

ROW_ID_PREFIX = "row_"


class DatasetGenerator:
    """Generates a batch of rubric-approved training rows, optionally seeded by a source dataset."""

    def __init__(
        self,
        num_rows: int,
        seed_dataset_name: str | None = None,
        unanswerable_fraction: float = config.UNANSWERABLE_FRACTION,
    ):
        self.num_rows = num_rows
        self.rows: list[TrainingRow] = []
        self.seed_dataset_name = seed_dataset_name or "generated"
        self.unanswerable_fraction = unanswerable_fraction
        self.seeds: list[SeedExample] = (
            SeedLoader(seed_dataset_name).load_seeds() if seed_dataset_name else []
        )
        self.row_ids = [f"{ROW_ID_PREFIX}{i:04d}" for i in range(num_rows)]
        # Reserved up front rather than sampled per row, so a run holds exactly
        # the requested share of unanswerable rows instead of roughly that many.
        self.unanswerable_ids = set(
            random.sample(self.row_ids, round(num_rows * unanswerable_fraction))
        )

    def generate_row(self, row_id: str) -> TrainingRow:
        """Generate one row that passes the rubric gate.

        Params:
            row_id: Identifier assigned to the row.

        Returns:
            The approved training row.

        Raises:
            RuntimeError: If every attempt is rejected or errors out.
        """
        for attempt in range(1, MAX_RETRIES_PER_ROW + 1):
            try:
                candidate = self._build_candidate(row_id)
                result = evaluate_row(candidate)
                if result.passed:
                    return candidate
                print(
                    f"  [{row_id}] attempt {attempt}/{MAX_RETRIES_PER_ROW} rejected: "
                    f"{', '.join(result.failure_reasons)}"
                )
            except Exception as error:
                print(f"  [{row_id}] attempt {attempt}/{MAX_RETRIES_PER_ROW} error: {error}")

        raise RuntimeError(f"Row {row_id} failed evaluation after {MAX_RETRIES_PER_ROW} attempts")

    def _build_candidate(self, row_id: str) -> TrainingRow:
        """Run one full generation pass without judging the result.

        Params:
            row_id: Identifier assigned to the row.

        Returns:
            An unevaluated candidate row.
        """
        seed = random.choice(self.seeds) if self.seeds else None
        if row_id in self.unanswerable_ids:
            return build_unanswerable_row(row_id, self.seed_dataset_name, seed)

        bundle = generate_instruction_bundle(seed)
        informative_chunks = build_informative_chunks(bundle)
        distractors = generate_distractors(
            instruction=bundle.instruction, informative_chunks=informative_chunks
        )
        search_pool = build_search_pool(informative_chunks, distractors)

        return TrainingRow(
            id=row_id,
            source=self.seed_dataset_name,
            seed=seed.instruction if seed else None,
            instruction=bundle.instruction,
            decomposition=build_decomposition(bundle, search_pool),
            search_pool=search_pool,
            rationale=generate_rationale(bundle.instruction, search_pool),
            response=generate_grounded_response(bundle.instruction, search_pool),
        )

    def generate_dataset(self) -> None:
        """Generate every row in parallel and store them on the instance in row id order."""
        results: dict[str, TrainingRow] = {}

        with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
            futures = {
                executor.submit(self.generate_row, row_id): row_id for row_id in self.row_ids
            }
            # as_completed yields on this thread only, so the counter needs no lock.
            for completed, future in enumerate(as_completed(futures), start=1):
                results[futures[future]] = future.result()
                if completed % 10 == 0:
                    print(f"Progress: {completed}/{self.num_rows} rows completed")

        self.rows = [results[row_id] for row_id in self.row_ids]

    def get_total_cost(self) -> tuple[float, float]:
        """Report what generation spent.

        Returns:
            Input and output token cost in dollars, summed across worker threads.
        """
        return get_aggregate_cost()
