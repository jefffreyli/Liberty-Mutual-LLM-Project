"""Builds rows whose search pool holds no informative chunk. The teacher writes a normal
instruction with its supporting paragraphs, the distractors are generated against those
paragraphs, and then the paragraphs are dropped, leaving a pool of topically adjacent results
that cannot complete the instruction and a response that declines to answer.
"""

from src.config.generation import NUM_UNANSWERABLE_DISTRACTORS
from src.generation.instruction import build_informative_chunks, generate_instruction_bundle
from src.generation.noise import build_search_pool, generate_distractors
from src.generation.prompts import (
    UNANSWERABLE_RATIONALE_PROMPT,
    UNANSWERABLE_RESPONSE_PROMPT,
)
from src.llm.client import get_llm_client
from src.render import render_pool
from src.schema import GroundedResponse, RationaleResponse, TrainingRow
from src.schema.seed import SeedExample


def build_unanswerable_row(row_id: str, source: str, seed: SeedExample | None) -> TrainingRow:
    """Build one candidate row whose pool is entirely distracting.

    The informative paragraphs are generated and then discarded, so the pool sits
    next to a real answer without containing it. The decomposition is left empty
    because its per hop answers are not recoverable from the pool.

    Params:
        row_id: Identifier assigned to the row.
        source: Seed dataset name recorded on the row.
        seed: Seed example to inspire the instruction, or None.

    Returns:
        An unevaluated candidate row with no informative chunk.
    """
    bundle = generate_instruction_bundle(seed)
    distractors = generate_distractors(
        instruction=bundle.instruction,
        informative_chunks=build_informative_chunks(bundle),
        n=NUM_UNANSWERABLE_DISTRACTORS,
    )
    # The informative chunks are dropped here; only the distractors reach the pool.
    search_pool = build_search_pool([], distractors)
    pool_text = render_pool(search_pool)

    return TrainingRow(
        id=row_id,
        source=source,
        seed=seed.instruction if seed else None,
        instruction=bundle.instruction,
        decomposition=[],
        search_pool=search_pool,
        rationale=_generate_rationale(bundle.instruction, pool_text),
        response=_generate_response(bundle.instruction, pool_text),
    )


def _generate_rationale(instruction: str, pool_text: str) -> str:
    """Generate the rationale that rejects every chunk in the pool.

    Params:
        instruction: The multi-hop instruction.
        pool_text: The rendered pool.

    Returns:
        The rationale text.
    """
    prompt = UNANSWERABLE_RATIONALE_PROMPT.format(
        instruction=instruction, search_pool_text=pool_text
    )
    return get_llm_client().generate(prompt, RationaleResponse).rationale


def _generate_response(instruction: str, pool_text: str) -> str:
    """Generate the response that declines to answer from the pool.

    Params:
        instruction: The multi-hop instruction.
        pool_text: The rendered pool.

    Returns:
        The response text.
    """
    prompt = UNANSWERABLE_RESPONSE_PROMPT.format(
        instruction=instruction, search_pool_text=pool_text
    )
    return get_llm_client().generate(prompt, GroundedResponse).response
