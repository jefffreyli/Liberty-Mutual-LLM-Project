"""Renders a search pool into the text that teacher and judge prompts embed. Kept separate from
src/training/format.py, which renders the pool the model itself reads: that one is a training
contract that cannot change without retraining, while these are free to change with the prompts.
"""

from src.schema import SearchResult


def render_pool(search_pool: list[SearchResult]) -> str:
    """Render a pool without revealing which chunks are informative.

    Params:
        search_pool: The chunks to render.

    Returns:
        The chunks as "[ID n] [title]: text" blocks.
    """
    return "\n\n".join(f"[ID {chunk.id}] [{chunk.title}]: {chunk.text}" for chunk in search_pool)


def render_labeled_pool(search_pool: list[SearchResult]) -> str:
    """Render a pool with its informative labels, for prompts that grade a finished row.

    Params:
        search_pool: The chunks to render.

    Returns:
        The chunks as "[ID n] [title] (informative=bool): text" blocks.
    """
    return "\n\n".join(
        f"[ID {chunk.id}] [{chunk.title}] (informative={chunk.is_informative}): {chunk.text}"
        for chunk in search_pool
    )
