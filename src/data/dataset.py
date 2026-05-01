from src.data.seed_loader import SeedLoader
import random
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.schema.seed import SeedExample

from ..config import MAX_RETRIES_PER_ROW
from ..schema import TrainingRow
from ..utils.llm_client import get_aggregate_cost
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
    def __init__(self, num_rows: int, seed_dataset_name: str | None = None):
        self.row_prefix = "row_"
        self.rows: list[TrainingRow] = []
        self.num_rows = num_rows
        self.seed_dataset_name = seed_dataset_name or "generated"
        self.seeds: list[SeedExample] = []
        if seed_dataset_name:
            seed_loader = SeedLoader(seed_dataset_name)
            self.seeds = seed_loader.load_seeds()

    def generate_row(self, row_id: str) -> TrainingRow:
        """
        Return a single training row that passes the rubric evaluation gate.
        If the row fails the evaluation gate, retry up to MAX_RETRIES_PER_ROW times.
        """
        for attempt in range(1, MAX_RETRIES_PER_ROW + 1):
            try:
                seed = random.choice(self.seeds) if self.seeds else None
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
                    source=self.seed_dataset_name,
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
            except Exception as e:
                print(f"  [{row_id}] attempt {attempt}/{MAX_RETRIES_PER_ROW} error: {e}")

        raise RuntimeError(
            f"Row {row_id} failed evaluation after {MAX_RETRIES_PER_ROW} attempts"
        )
    
    def generate_dataset(self) -> None:
        """Generate the full dataset in parallel using a thread pool."""
        num_workers = config.NUM_WORKERS
        completed_count = 0
        lock = threading.Lock()

        row_ids = [f"{self.row_prefix}{i:04d}" for i in range(self.num_rows)]

        # results dict preserves insertion order by row_id
        results: dict[str, TrainingRow] = {}

        def _generate(row_id: str) -> tuple[str, TrainingRow]:
            return row_id, self.generate_row(row_id)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_generate, row_id): row_id for row_id in row_ids}
            for future in as_completed(futures):
                row_id, row = future.result()
                results[row_id] = row
                with lock:
                    completed_count += 1
                    if completed_count % 10 == 0:
                        print(f"Progress: {completed_count}/{self.num_rows} rows completed")

        # Restore original ordering by row_id
        self.rows = [results[row_id] for row_id in row_ids]

    def get_total_cost(self) -> tuple[float, float]:
        """Return aggregate input/output token costs across all worker threads."""
        return get_aggregate_cost()
    
    def convert_to_chatml_format(self) -> list[dict]:
        """Convert the dataset's rows into a list of ChatML-style examples."""
        system_prompt = (
            "You are a helpful assistant. Use only the provided search "
            "results to answer the instruction. First explain which search "
            "results are relevant and why, then provide the final grounded "
            "response."
        )
        chatml_data = []
        for row in self.rows:
            search_pool_json = json.dumps(
                [chunk.model_dump() for chunk in row.search_pool],
                ensure_ascii=False,
                indent=2,
            )
            chatml_data.append({
                "id": row.id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Instruction:\n{row.instruction}\n\n"
                            f"Search results:\n{search_pool_json}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Rationale:\n{row.rationale}\n\n"
                            f"Response:\n{row.response}"
                        ),
                    },
                ],
            })
        return chatml_data
