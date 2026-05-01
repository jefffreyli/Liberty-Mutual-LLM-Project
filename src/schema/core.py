from pydantic import BaseModel


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
    """Complete SAIL training example."""

    id: str
    source: str = "generated"  # seed dataset name, or "generated" if no seed was used
    instruction: str
    decomposition: list[DecompositionStep]
    search_pool: list[SearchResult]
    rationale: str
    response: str
