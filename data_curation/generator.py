"""SAIL training row generation: question, decomposition, search pool, rationale, response."""

from data_curation.llm import generate
from data_curation.noise import generate_hard_negatives, build_search_pool
from data_curation.schemas import (
    SearchResult,
    DecompositionStep,
    TrainingRow,
    QuestionGenerationResponse,
    RationaleResponse,
    GroundedResponse,
)

QUESTION_PROMPT = """
You are an expert insurance knowledge engineer. Generate a challenging multi-hop question that requires reasoning across multiple insurance concepts.

The question must:
- Require at least 2 reasoning steps (hops) to answer
- Cover real insurance topics (e.g. underwriting, claims, policy terms, regulations, actuarial concepts)
- Have a clear, factual answer

Provide:
1. The multi-hop question
2. The decomposition into single-hop sub-questions (each with its answer)
3. Informative paragraphs (2-4) that contain the information needed to answer each sub-question
4. The final answer"""

RATIONALE_PROMPT = """
You are analyzing a search pool for a multi-hop insurance QA task.

Question: {question}

Search pool:
{search_pool_text}

Informative paragraph IDs: {informative_ids}

Write a rationale that:
1. Identifies which search result IDs are informative and which are distracting
2. Explains WHY each distracting chunk is misleading (e.g. outdated, misattributed, partial truth)
3. Traces the reasoning chain through the informative paragraphs to reach the answer"""

RESPONSE_PROMPT = """
You are answering an insurance question using ONLY the provided informative paragraphs. Do not use any outside knowledge.

Question: {question}

Informative paragraphs:
{informative_text}

Write a comprehensive response grounded entirely in the informative paragraphs above. Cite specific details from the paragraphs."""


def generate_row(row_id: str) -> TrainingRow:
    """Produce one SAIL training row: question, decomposition, search pool, rationale, grounded response."""
    qa = generate(QUESTION_PROMPT, QuestionGenerationResponse)

    informative_chunks = [
        SearchResult(id=-1, title=p.title, text=p.text, is_informative=True)
        for p in qa.informative_paragraphs
    ]

    hard_negatives = generate_hard_negatives(
        topic=qa.question, informative_chunks=informative_chunks
    )
    search_pool = build_search_pool(informative_chunks, hard_negatives)

    decomposition = []
    for i, step in enumerate(qa.decomposition):
        informative_for_step = next(
            (s for s in search_pool if s.is_informative and s.title == qa.informative_paragraphs[min(i, len(qa.informative_paragraphs) - 1)].title),
            search_pool[0],
        )
        decomposition.append(
            DecompositionStep(
                id=i,
                question=step.question,
                answer=step.answer,
                support_paragraph_id=informative_for_step.id,
            )
        )

    informative_ids = [s.id for s in search_pool if s.is_informative]
    search_pool_text = "\n\n".join(
        f"[ID {s.id}] [{s.title}]: {s.text}" for s in search_pool
    )
    rationale = generate(
        RATIONALE_PROMPT.format(
            question=qa.question,
            search_pool_text=search_pool_text,
            informative_ids=informative_ids,
        ),
        RationaleResponse,
    ).rationale

    informative_text = "\n\n".join(
        f"[{s.title}]: {s.text}" for s in search_pool if s.is_informative
    )
    response = generate(
        RESPONSE_PROMPT.format(question=qa.question, informative_text=informative_text),
        GroundedResponse,
    ).response

    return TrainingRow(
        id=row_id,
        instruction=qa.question,
        decomposition=decomposition,
        search_pool=search_pool,
        rationale=rationale,
        response=response,
    )


def generate_dataset(n: int) -> list[TrainingRow]:
    """Generate n SAIL training rows."""
    rows = []
    for i in range(n):
        row = generate_row(row_id=f"sail_{i:04d}")
        rows.append(row)
    return rows
