"""Core schemas describing a finished training row: the multi-hop instruction, the single-hop
decomposition, and the search pool of informative and distracting chunks it was built from.
"""

from pydantic import BaseModel

# Placeholder id carried by a chunk until build_search_pool shuffles the pool
# and assigns the sequential ids that the model cites.
UNASSIGNED_ID = -1


class SearchResult(BaseModel):
    """A single chunk in the search pool; informative or distracting."""

    id: int
    title: str
    text: str
    is_informative: bool


class DecompositionStep(BaseModel):
    """Single-hop sub-instruction with its output and supporting paragraph reference."""

    id: int
    instruction: str
    answer: str
    support_paragraph_id: int


class TrainingRow(BaseModel):
    """Complete multi-hop training example."""

    id: str
    source: str = "generated"  # seed dataset name, or "generated" if no seed was used
    seed: str | None = None  # seed instruction text used to inspire this row, or None
    instruction: str
    decomposition: list[DecompositionStep]
    search_pool: list[SearchResult]
    rationale: str
    response: str
