"""SAIL training row generation with rubric-based quality gate."""

from ..config import MAX_RETRIES_PER_ROW
from ..schema import (
    DecompositionStep,
    GroundedResponse,
    InstructionGenerationResponse,
    RationaleResponse,
    SearchResult,
    TrainingRow,
)
from ..utils.llm_client import generate
from .evaluator import evaluate_row
from .noise import build_search_pool, generate_distractors
from .prompts import GENERAL_INSTRUCTION_PROMPT, RATIONALE_PROMPT, RESPONSE_PROMPT


def _generate_instruction_bundle() -> InstructionGenerationResponse:
    """Generate a multi-hop instruction with decomposition and informative paragraphs."""
    return generate(GENERAL_INSTRUCTION_PROMPT, InstructionGenerationResponse)


def _build_informative_chunks(qa: InstructionGenerationResponse) -> list[SearchResult]:
    """Convert LLM-generated informative paragraphs into SearchResult objects."""
    return [
        SearchResult(id=-1, title=p.title, text=p.text, is_informative=True)
        for p in qa.informative_paragraphs
    ]


def _build_decomposition(
    qa: InstructionGenerationResponse,
    search_pool: list[SearchResult],
) -> list[DecompositionStep]:
    """Map decomposition steps to supporting informative paragraphs in search pool."""
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


def _generate_rationale(instruction: str, search_pool: list[SearchResult]) -> str:
    """Generate rationale identifying informative vs distracting chunks."""
    informative_ids = [s.id for s in search_pool if s.is_informative]
    search_pool_text = "\n\n".join(
        f"[ID {s.id}] [{s.title}]: {s.text}" for s in search_pool
    )
    return generate(
        RATIONALE_PROMPT.format(
            instruction=instruction,
            search_pool_text=search_pool_text,
            informative_ids=informative_ids,
        ),
        RationaleResponse,
    ).rationale


def _generate_grounded_response(instruction: str, search_pool: list[SearchResult]) -> str:
    """Generate a response grounded only in the informative paragraphs."""
    informative_text = "\n\n".join(
        f"[{s.title}]: {s.text}" for s in search_pool if s.is_informative
    )
    return generate(
        RESPONSE_PROMPT.format(instruction=instruction, informative_text=informative_text),
        GroundedResponse,
    ).response


def _generate_candidate_row(row_id: str) -> TrainingRow:
    """Build one complete candidate training row (before evaluation)."""
    qa = _generate_instruction_bundle()
    informative_chunks = _build_informative_chunks(qa)

    distractors = generate_distractors(
        instruction=qa.instruction, informative_chunks=informative_chunks
    )
    search_pool = build_search_pool(informative_chunks, distractors)
    decomposition = _build_decomposition(qa, search_pool)
    rationale = _generate_rationale(qa.instruction, search_pool)
    response = _generate_grounded_response(qa.instruction, search_pool)

    return TrainingRow(
        id=row_id,
        instruction=qa.instruction,
        decomposition=decomposition,
        search_pool=search_pool,
        rationale=rationale,
        response=response,
    )


def generate_row(row_id: str) -> TrainingRow:
    """
    Generate one training row that passes the rubric evaluation gate.
    If the row fails the evaluation gate, retry up to MAX_RETRIES_PER_ROW times.
    """
    for attempt in range(1, MAX_RETRIES_PER_ROW + 1):
        candidate = _generate_candidate_row(row_id)
        result = evaluate_row(candidate)

        if result.passed:
            return candidate

        print(
            f"  [{row_id}] attempt {attempt}/{MAX_RETRIES_PER_ROW} rejected: "
            f"{', '.join(result.failure_reasons)}"
        )

    raise RuntimeError(
        f"Row {row_id} failed evaluation after {MAX_RETRIES_PER_ROW} attempts"
    )


def generate_dataset(n: int) -> list[TrainingRow]:
    """Generate n SAIL training rows, each passing the rubric quality gate."""
    rows: list[TrainingRow] = []
    for i in range(n):
        rows.append(generate_row(row_id=f"sail_{i:04d}"))
    return rows
