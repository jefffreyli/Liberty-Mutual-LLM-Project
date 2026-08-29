"""Core schemas describing a finished training row: the multi-hop instruction, the single-hop
decomposition, and the search pool of informative and distracting chunks it was built from.
"""

from pydantic import BaseModel, model_validator

# Placeholder id carried by a chunk until build_search_pool shuffles the pool
# and assigns the sequential ids that the model cites.
UNASSIGNED_ID = -1


class SearchResult(BaseModel):
    """A single chunk in the search pool: informative, plainly distracting, or contradictory.

    A contradictory chunk states a false version of a fact the informative chunks
    carry, so the model has to prefer the mutually consistent gold cluster rather
    than merely judging topical relevance. It is never informative.
    """

    id: int
    title: str
    text: str
    is_informative: bool
    is_contradictory: bool = False

    @model_validator(mode="after")
    def _check_exclusive(self) -> "SearchResult":
        """Reject a chunk marked both informative and contradictory.

        Returns:
            The validated chunk.

        Raises:
            ValueError: If the two flags are set together, which would make the
                gold set self contradictory.
        """
        if self.is_informative and self.is_contradictory:
            raise ValueError(f"chunk {self.id} cannot be both informative and contradictory")
        return self


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

    @property
    def contradictory_ids(self) -> list[int]:
        """Collect the IDs of chunks that state a false version of a gold fact.

        Returns:
            The contradictory chunk IDs, in pool order.
        """
        return [chunk.id for chunk in self.search_pool if chunk.is_contradictory]

    @property
    def is_answerable(self) -> bool:
        """Whether the search pool can complete the instruction.

        Returns:
            True when at least one chunk is informative. An unanswerable row has
            a pool of distractors only, an empty decomposition, and a response
            that declines to answer.
        """
        return any(chunk.is_informative for chunk in self.search_pool)
