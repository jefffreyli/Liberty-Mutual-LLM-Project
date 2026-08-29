"""Generates the adversarial half of a training row: distractor paragraphs that shift one
parameter of a gold paragraph, and contradictory paragraphs that state a false version of a gold
fact, then merges everything with the informative chunks into a shuffled search pool with
sequential ids.
"""

import random

from src.config.generation import NUM_CONTRADICTORY, NUM_DISTRACTORS
from src.generation.prompts import CONTRADICTORY_PROMPT, DISTRACTOR_PROMPT
from src.schema import (
    UNASSIGNED_ID,
    ContradictoryResponse,
    DistractorResponse,
    SearchResult,
)
from src.llm.client import get_llm_client


def generate_distractors(
    instruction: str,
    informative_chunks: list[SearchResult],
    n: int = NUM_DISTRACTORS,
) -> list[SearchResult]:
    """Generate distractor paragraphs adjacent to the informative ones.

    Params:
        instruction: The multi-hop instruction.
        informative_chunks: The chunks the distractors should sit next to.
        n: How many distractors to request.

    Returns:
        Distractor chunks awaiting their pool ids.
    """
    prompt = DISTRACTOR_PROMPT.format(
        instruction=instruction, informative_text=_render(informative_chunks), n=n
    )
    result = get_llm_client().generate(prompt, DistractorResponse)
    return [
        SearchResult(id=UNASSIGNED_ID, title=p.title, text=p.text, is_informative=False)
        for p in result.paragraphs
    ]


def generate_contradictory(
    instruction: str,
    informative_chunks: list[SearchResult],
    n: int = NUM_CONTRADICTORY,
) -> list[SearchResult]:
    """Generate paragraphs that state a false version of a gold fact.

    These are what force the model to weigh truth as well as relevance: each one
    is on topic for the instruction and wrong, so only its disagreement with the
    mutually consistent informative chunks marks it out.

    Params:
        instruction: The multi-hop instruction.
        informative_chunks: The gold chunks whose facts are contradicted.
        n: How many contradictory paragraphs to request.

    Returns:
        Contradictory chunks awaiting their pool ids.
    """
    if n <= 0:
        return []
    prompt = CONTRADICTORY_PROMPT.format(
        instruction=instruction, informative_text=_render(informative_chunks), n=n
    )
    result = get_llm_client().generate(prompt, ContradictoryResponse)
    return [
        SearchResult(
            id=UNASSIGNED_ID,
            title=p.title,
            text=p.text,
            is_informative=False,
            is_contradictory=True,
        )
        for p in result.paragraphs
    ]


def _render(chunks: list[SearchResult]) -> str:
    """Render chunks as the titled blocks the noise prompts embed.

    Params:
        chunks: The chunks to render.

    Returns:
        One titled block per chunk.
    """
    return "\n\n".join(f"[{c.title}]: {c.text}" for c in chunks)


def build_search_pool(
    informative_chunks: list[SearchResult],
    distractors: list[SearchResult],
    contradictory: list[SearchResult] | None = None,
) -> list[SearchResult]:
    """Merge and shuffle every chunk class, then assign the ids the model will cite.

    Params:
        informative_chunks: The gold chunks.
        distractors: The neighboring-case chunks.
        contradictory: Chunks stating a false version of a gold fact, if any.

    Returns:
        The pool in shuffled order with sequential ids.
    """
    pool = informative_chunks + distractors + list(contradictory or [])
    random.shuffle(pool)
    return [chunk.model_copy(update={"id": i}) for i, chunk in enumerate(pool)]
