"""Pydantic schemas for SAIL training data and LLM response parsing."""

from pydantic import BaseModel

class SearchResult(BaseModel):
    """A single chunk in the search pool; informative or distracting."""

    id: int
    title: str
    text: str
    is_informative: bool


class DecompositionStep(BaseModel):
    """Single-hop sub-question with its answer and supporting paragraph reference."""

    id: int
    question: str
    answer: str
    support_paragraph_id: int


class TrainingRow(BaseModel):
    """Complete SAIL training example: instruction, decomposition, search pool, rationale, response."""

    id: str
    instruction: str
    decomposition: list[DecompositionStep]
    search_pool: list[SearchResult]
    rationale: str
    response: str


# Intermediate models for structured LLM outputs

class InformativeParagraph(BaseModel):
    """Paragraph containing correct information for answering a sub-question."""

    title: str
    text: str


class GeneratedDecompositionStep(BaseModel):
    """Single-hop question and answer from the LLM."""

    question: str
    answer: str


class QuestionGenerationResponse(BaseModel):
    """LLM output: multi-hop question, decomposition, informative paragraphs, final answer."""

    question: str
    decomposition: list[GeneratedDecompositionStep]
    informative_paragraphs: list[InformativeParagraph]
    answer: str


class HardNegativeParagraph(BaseModel):
    """Adversarial distractor paragraph (plausible but wrong)."""

    title: str
    text: str


class HardNegativeResponse(BaseModel):
    """LLM output: list of hard-negative paragraphs."""

    paragraphs: list[HardNegativeParagraph]


class RationaleResponse(BaseModel):
    """LLM output: rationale identifying informative vs distracting chunks."""

    rationale: str


class GroundedResponse(BaseModel):
    """LLM output: response grounded only in informative paragraphs."""

    response: str
