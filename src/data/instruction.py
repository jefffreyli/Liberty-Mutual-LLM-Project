"""Training row generation with rubric-based quality gate."""

from ..schema import (
    DecompositionStep,
    GroundedResponse,
    InstructionGenerationResponse,
    RationaleResponse,
    SearchResult,
)
from ..utils.llm_client import get_llm_client
from .prompts import GENERAL_INSTRUCTION_PROMPT, RATIONALE_PROMPT, RESPONSE_PROMPT

def generate_instruction_bundle() -> InstructionGenerationResponse:
    """
    Generate a multi-hop instruction with decomposition and informative paragraphs.
    """
    client = get_llm_client()
    return client.generate(GENERAL_INSTRUCTION_PROMPT, InstructionGenerationResponse)

def build_informative_chunks(qa: InstructionGenerationResponse) -> list[SearchResult]:
    """
    Convert LLM-generated informative paragraphs into SearchResult objects.
    """
    return [
        SearchResult(id=-1, title=p.title, text=p.text, is_informative=True)
        for p in qa.informative_paragraphs
    ]

def build_decomposition(qa: InstructionGenerationResponse, search_pool: list[SearchResult]) -> list[DecompositionStep]:
    """
    Map decomposition steps to supporting informative paragraphs in search pool.
    """
    informative_pool = [s for s in search_pool if s.is_informative]
    steps: list[DecompositionStep] = []

    for i, step in enumerate(qa.decomposition):
        para_index = min(i, len(qa.informative_paragraphs) - 1)
        target_title = qa.informative_paragraphs[para_index].title

        match = next((s for s in informative_pool if s.title == target_title), None)
        if match is None:
            raise ValueError(
                f"Decomposition step {i} references paragraph '{target_title}' "
                f"which was not found in the search pool"
            )

        steps.append(
            DecompositionStep(
                id=i,
                instruction=step.instruction,
                answer=step.answer,
                support_paragraph_id=match.id,
            )
        )

    return steps

def generate_rationale(
    instruction: str,
    search_pool: list[SearchResult],
) -> str:
    """Generate rationale identifying informative vs distracting chunks."""
    informative_ids = [s.id for s in search_pool if s.is_informative]
    search_pool_text = "\n\n".join(
        f"[ID {s.id}] [{s.title}]: {s.text}" for s in search_pool
    )
    client = get_llm_client()
    return client.generate(
        RATIONALE_PROMPT.format(
            instruction=instruction,
            search_pool_text=search_pool_text,
            informative_ids=informative_ids,
        ),
        RationaleResponse,
    ).rationale


def generate_grounded_response(
    instruction: str,
    search_pool: list[SearchResult],
) -> str:
    """Generate a response grounded only in the informative paragraphs."""
    informative_text = "\n\n".join(
        f"[{s.title}]: {s.text}" for s in search_pool if s.is_informative
    )
    client = get_llm_client()
    return client.generate(
        RESPONSE_PROMPT.format(instruction=instruction, informative_text=informative_text),
        GroundedResponse,
    ).response



