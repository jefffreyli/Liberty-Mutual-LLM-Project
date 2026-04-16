from src.data.seed_loader import SeedLoader
import random

from ..config import MAX_RETRIES_PER_ROW
from ..schema import TrainingRow
from ..utils.llm_client import get_llm_client
from .evaluator import evaluate_row
from .instruction import (
    build_decomposition,
    build_informative_chunks,
    generate_grounded_response,
    generate_instruction_bundle,
    generate_rationale,
)
from .noise import build_search_pool, generate_distractors
from src import config

class Dataset:
    def __init__(self, num_rows: int):
        self.row_prefix = "row_"
        self.rows: list[TrainingRow] = []
        self.num_rows = num_rows
        dataset_name = config.SEED_DATASET_NAME
        seed_loader = SeedLoader(dataset_name)
        self.seeds = seed_loader.load_seeds()
        if not self.seeds:
            raise ValueError(f"No seeds loaded from dataset '{dataset_name}'")

    def generate_row(self, row_id: str) -> TrainingRow:
        """
        Return a single training row that passes the rubric evaluation gate.
        If the row fails the evaluation gate, retry up to MAX_RETRIES_PER_ROW times.
        """
        for attempt in range(1, MAX_RETRIES_PER_ROW + 1):
            seed = random.choice(self.seeds)
            qa = generate_instruction_bundle(seed)
            informative_chunks = build_informative_chunks(qa)

            distractors = generate_distractors(
                instruction=qa.instruction,
                informative_chunks=informative_chunks,
            )
            search_pool = build_search_pool(informative_chunks, distractors)
            decomposition = build_decomposition(qa, search_pool)
            rationale = generate_rationale(qa.instruction, search_pool)
            response = generate_grounded_response(qa.instruction, search_pool)

            candidate = TrainingRow(
                id=row_id,
                instruction=qa.instruction,
                decomposition=decomposition,
                search_pool=search_pool,
                rationale=rationale,
                response=response,
            )
            result = evaluate_row(candidate)

            if result.passed:
                return candidate

            print(
                f"  [{row_id}] attempt {attempt}/{MAX_RETRIES_PER_ROW} rejected: "
                f"{', '.join(result.failure_reasons)}"
            )

        raise RuntimeError(
            f"Row {row_id} failed evaluation after {MAX_RETRIES_PER_ROW} attempts"
        )
    
    def generate_dataset(self) -> None:
        """Generate the full dataset of rows."""
        for i in range(self.num_rows):
            row = self.generate_row(row_id=f"{self.row_prefix}{i:04d}")
            self.rows.append(row)

    def get_total_cost(self) -> tuple[float, float]:
        """Return aggregate input/output token costs for this dataset run."""
        llm_client = get_llm_client()
        return llm_client.get_total_cost()

