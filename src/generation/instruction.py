"""Prompts the teacher model for the informative half of a training row: the multi-hop
instruction with its decomposition and supporting paragraphs, the rationale that separates
informative from distracting chunks, and the grounded response.
"""

from src.generation.prompts import (
    GENERAL_INSTRUCTION_PROMPT,
    NO_SEED_INSTRUCTION_BLOCK,
    RATIONALE_PROMPT,
    RESPONSE_PROMPT,
    SEED_EXAMPLE_BLOCK,
)
from src.schema import (
    UNASSIGNED_ID,
    DecompositionStep,
    GroundedResponse,
    InstructionGenerationResponse,
    RationaleResponse,
    SearchResult,
)
from src.schema.seed import SeedExample
from src.llm.client import get_llm_client
from src.render import render_pool


def _format_seed_block(seed: SeedExample | None) -> str:
    """Render the seed example that primes the instruction prompt.

    Params:
        seed: Seed example to imitate in style, or None to generate from scratch.

    Returns:
        The prompt block describing the seed.
    """
    if seed is None:
        return NO_SEED_INSTRUCTION_BLOCK

    decomposition = (
        "\n".join(
            f"  {i + 1}. Q: {step.question} A: {step.answer}"
            for i, step in enumerate(seed.question_decomposition)
        )
        or "  (none)"
    )
    return SEED_EXAMPLE_BLOCK.format(
        seed_instruction=seed.instruction,
        seed_paragraphs=seed.paragraphs or "(none)",
        seed_decomposition=decomposition,
        seed_output=seed.output,
    )


def generate_instruction_bundle(seed: SeedExample | None = None) -> InstructionGenerationResponse:
    """Generate a multi-hop instruction with its decomposition and informative paragraphs.

    Params:
        seed: Seed example to take inspiration from, or None to generate from scratch.

    Returns:
        The teacher model's instruction bundle.
    """
    prompt = GENERAL_INSTRUCTION_PROMPT.format(seed_block=_format_seed_block(seed))
    return get_llm_client().generate(prompt, InstructionGenerationResponse)


def build_informative_chunks(bundle: InstructionGenerationResponse) -> list[SearchResult]:
    """Convert the generated informative paragraphs into search pool chunks.

    Params:
        bundle: The generated instruction bundle.

    Returns:
        Informative chunks awaiting their pool ids.
    """
    return [
        SearchResult(id=UNASSIGNED_ID, title=p.title, text=p.text, is_informative=True)
        for p in bundle.informative_paragraphs
    ]


def build_decomposition(
    bundle: InstructionGenerationResponse, search_pool: list[SearchResult]
) -> list[DecompositionStep]:
    """Link each decomposition step to the pool id of the paragraph that supports it.

    Params:
        bundle: The generated instruction bundle.
        search_pool: The shuffled pool with ids already assigned.

    Returns:
        Decomposition steps carrying their support paragraph id.

    Raises:
        ValueError: If the bundle has no informative paragraphs, or a step's
            paragraph is missing from the pool.
    """
    if not bundle.informative_paragraphs:
        raise ValueError("Instruction bundle has no informative paragraphs")

    pool_id_by_title = {chunk.title: chunk.id for chunk in search_pool if chunk.is_informative}
    steps: list[DecompositionStep] = []

    for i, step in enumerate(bundle.decomposition):
        # The model may return fewer paragraphs than steps, so later steps fall
        # back to the last paragraph rather than failing the whole row.
        paragraph = bundle.informative_paragraphs[min(i, len(bundle.informative_paragraphs) - 1)]
        if paragraph.title not in pool_id_by_title:
            raise ValueError(
                f"Decomposition step {i} references paragraph '{paragraph.title}' "
                f"which was not found in the search pool"
            )
        steps.append(
            DecompositionStep(
                id=i,
                instruction=step.instruction,
                answer=step.answer,
                support_paragraph_id=pool_id_by_title[paragraph.title],
            )
        )

    return steps


def generate_rationale(instruction: str, search_pool: list[SearchResult]) -> str:
    """Generate the rationale that separates informative chunks from distractors.

    Params:
        instruction: The multi-hop instruction.
        search_pool: The full pool, informative chunks included.

    Returns:
        The rationale text.
    """
    prompt = RATIONALE_PROMPT.format(
        instruction=instruction,
        search_pool_text=render_pool(search_pool),
        informative_ids=[s.id for s in search_pool if s.is_informative],
    )
    return get_llm_client().generate(prompt, RationaleResponse).rationale


def generate_grounded_response(instruction: str, search_pool: list[SearchResult]) -> str:
    """Generate the response, shown only the informative chunks so it stays grounded.

    Params:
        instruction: The multi-hop instruction.
        search_pool: The full pool; distractors are withheld from the prompt.

    Returns:
        The response text.
    """
    informative_text = "\n\n".join(
        f"[{s.title}]: {s.text}" for s in search_pool if s.is_informative
    )
    prompt = RESPONSE_PROMPT.format(instruction=instruction, informative_text=informative_text)
    return get_llm_client().generate(prompt, GroundedResponse).response
