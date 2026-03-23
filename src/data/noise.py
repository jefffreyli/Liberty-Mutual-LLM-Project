"""Neighboring-concept distractor generation for the search pool."""

import random

from ..config import NUM_DISTRACTORS
from ..schema import DistractorResponse, SearchResult
from ..utils.llm_client import generate
from .prompts import DISTRACTOR_PROMPT


def generate_distractors(
    instruction: str, informative_chunks: list[SearchResult], n: int = NUM_DISTRACTORS
) -> list[SearchResult]:
    """Generate n neighboring-concept distractor paragraphs."""
    informative_text = "\n\n".join(
        f"[{c.title}]: {c.text}" for c in informative_chunks
    )
    prompt = DISTRACTOR_PROMPT.format(
        instruction=instruction, informative_text=informative_text, n=n
    )
    result = generate(prompt, DistractorResponse)
    return [
        SearchResult(id=-1, title=p.title, text=p.text, is_informative=False)
        for p in result.paragraphs
    ]


def build_search_pool(
    informative_chunks: list[SearchResult], distractors: list[SearchResult]
) -> list[SearchResult]:
    """Merge informative and distractor chunks, shuffle, and assign sequential IDs."""
    pool = informative_chunks + distractors
    random.shuffle(pool)
    return [chunk.model_copy(update={"id": i}) for i, chunk in enumerate(pool)]
