"""Adversarial hard-negative generation for the search pool."""

import random
from data_curation.llm import generate
from data_curation.schemas import SearchResult, HardNegativeResponse

HARD_NEGATIVE_PROMPT = """
You are generating adversarial distractor paragraphs for a search-augmented QA training dataset in the insurance domain.

Topic: {topic}

Here are the informative paragraphs the question is based on:
{informative_text}

Generate exactly {n} distracting paragraphs. Each should be:
- Topically related to the informative paragraphs (same insurance domain/concepts)
- Plausible-sounding but WRONG in a subtle way (outdated figures, misattributed policies, confused terminology, partial truths)
- Designed to trick a model into citing them instead of the informative paragraphs

Each paragraph needs a realistic title and body text.
"""


def generate_hard_negatives(
    topic: str, informative_chunks: list[SearchResult], n: int = 8
) -> list[SearchResult]:
    """Generate n adversarial distracting paragraphs (topically related but subtly wrong)."""
    informative_text = "\n\n".join(
        f"[{c.title}]: {c.text}" for c in informative_chunks
    )
    prompt = HARD_NEGATIVE_PROMPT.format(topic=topic, informative_text=informative_text, n=n)
    result = generate(prompt, HardNegativeResponse)
    return [
        SearchResult(id=-1, title=p.title, text=p.text, is_informative=False)
        for p in result.paragraphs
    ]


def build_search_pool(
    informative_chunks: list[SearchResult], hard_negatives: list[SearchResult]
) -> list[SearchResult]:
    """Merge informative and distracting chunks, shuffle, and assign sequential IDs."""
    pool = informative_chunks + hard_negatives
    random.shuffle(pool)
    return [chunk.model_copy(update={"id": i}) for i, chunk in enumerate(pool)]
