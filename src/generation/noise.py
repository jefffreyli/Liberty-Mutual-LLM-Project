"""Generates the adversarial half of a training row: neighboring-concept distractor paragraphs
that a retriever would plausibly return, then merges them with the informative chunks into a
shuffled search pool with sequential ids.
"""

import random

from src.config.generation import NUM_DISTRACTORS
from src.generation.prompts import DISTRACTOR_PROMPT
from src.schema import UNASSIGNED_ID, DistractorResponse, SearchResult
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
    informative_text = "\n\n".join(f"[{c.title}]: {c.text}" for c in informative_chunks)
    prompt = DISTRACTOR_PROMPT.format(
        instruction=instruction, informative_text=informative_text, n=n
    )
    result = get_llm_client().generate(prompt, DistractorResponse)
    return [
        SearchResult(id=UNASSIGNED_ID, title=p.title, text=p.text, is_informative=False)
        for p in result.paragraphs
    ]


def build_search_pool(
    informative_chunks: list[SearchResult], distractors: list[SearchResult]
) -> list[SearchResult]:
    """Merge and shuffle the chunks, then assign the ids the model will cite.

    Params:
        informative_chunks: The gold chunks.
        distractors: The adversarial chunks.

    Returns:
        The pool in shuffled order with sequential ids.
    """
    pool = informative_chunks + distractors
    random.shuffle(pool)
    return [chunk.model_copy(update={"id": i}) for i, chunk in enumerate(pool)]
