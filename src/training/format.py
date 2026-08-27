"""Defines the single chat format shared by SFT targets, RL prompts, and RL grading, so that
the model is graded on exactly the format it was taught: an "Informative IDs" line that names
the gold search results, followed by a rationale and a grounded response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.schema import SearchResult, TrainingRow

SYSTEM_PROMPT = """You are a research assistant answering multi-hop instructions from a noisy search pool.

Some search results are informative; others are distracting, outdated, or contradict the informative ones. A search result is not trustworthy just because it was retrieved.

Answer in exactly this format:

Informative IDs: [comma-separated IDs of the informative search results]
Rationale:
<which search results are informative and why, and why the remaining ones are distracting>
Response:
<the answer to the instruction, grounded only in the informative search results>"""

RATIONALE_HEADER = "Rationale:"
RESPONSE_HEADER = "Response:"
INFORMATIVE_IDS_HEADER = "Informative IDs:"

_IDS_PATTERN = re.compile(rf"{INFORMATIVE_IDS_HEADER}\s*\[([^\]]*)\]", re.IGNORECASE)
_RATIONALE_PATTERN = re.compile(
    rf"^{RATIONALE_HEADER}\s*(.*?)(?=^{RESPONSE_HEADER})",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_RESPONSE_PATTERN = re.compile(
    rf"^{RESPONSE_HEADER}\s*(.*)", re.IGNORECASE | re.MULTILINE | re.DOTALL
)


@dataclass(frozen=True)
class ParsedAnswer:
    """A model answer split into cited IDs, rationale, and response.

    `cited_ids` is None when the header is missing or unparseable, which is
    different from an empty list (the model claimed nothing is informative).
    """

    cited_ids: list[int] | None
    rationale: str
    response: str

    @property
    def is_well_formed(self) -> bool:
        """Whether the answer has both cited IDs and a non-empty response.

        Returns:
            True if the answer can be graded.
        """
        return self.cited_ids is not None and bool(self.response.strip())


def render_search_pool(search_pool: list[SearchResult]) -> str:
    """Render the search pool as the model sees it.

    Params:
        search_pool: Gold and distractor chunks for one row.

    Returns:
        The chunks as "[ID n] title" blocks, without the is_informative label
        that the model is supposed to predict.
    """
    return "\n\n".join(f"[ID {chunk.id}] {chunk.title}\n{chunk.text}" for chunk in search_pool)


def build_user_prompt(row: TrainingRow) -> str:
    """Build the user turn for a row.

    Params:
        row: The training row.

    Returns:
        The instruction followed by the rendered search pool.
    """
    return (
        f"Instruction:\n{row.instruction}\n\nSearch results:\n{render_search_pool(row.search_pool)}"
    )


def gold_informative_ids(row: TrainingRow) -> list[int]:
    """Collect the IDs of the row's informative chunks.

    Params:
        row: The training row.

    Returns:
        The gold search result IDs, in pool order.
    """
    return [chunk.id for chunk in row.search_pool if chunk.is_informative]


def build_assistant_target(row: TrainingRow) -> str:
    """Build the SFT target from the teacher rationale and response.

    Params:
        row: The training row.

    Returns:
        The assistant turn in the shared answer format.
    """
    ids = ", ".join(str(i) for i in gold_informative_ids(row))
    return (
        f"{INFORMATIVE_IDS_HEADER} [{ids}]\n"
        f"{RATIONALE_HEADER}\n{row.rationale.strip()}\n"
        f"{RESPONSE_HEADER}\n{row.response.strip()}"
    )


def row_to_messages(row: TrainingRow) -> list[dict[str, str]]:
    """Convert a training row into a ChatML conversation.

    Params:
        row: The training row.

    Returns:
        System, user, and assistant messages.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
        {"role": "assistant", "content": build_assistant_target(row)},
    ]


def parse_answer(text: str) -> ParsedAnswer:
    """Parse a model answer into its three parts.

    Params:
        text: Raw model completion.

    Returns:
        The parsed answer. Never raises, since malformed answers are expected
        during RL.
    """
    ids_match = _IDS_PATTERN.search(text)
    cited_ids = (
        sorted({int(tok) for tok in re.findall(r"\d+", ids_match.group(1))}) if ids_match else None
    )
    rationale_match = _RATIONALE_PATTERN.search(text)
    response_match = _RESPONSE_PATTERN.search(text)
    return ParsedAnswer(
        cited_ids=cited_ids,
        rationale=rationale_match.group(1).strip() if rationale_match else "",
        response=response_match.group(1).strip() if response_match else "",
    )
