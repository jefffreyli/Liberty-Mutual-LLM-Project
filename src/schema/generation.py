"""Intermediate schemas for generation-stage LLM calls."""

from pydantic import BaseModel


class InformativeParagraph(BaseModel):
    """Paragraph containing correct information for executing a sub-instruction."""

    title: str
    text: str


class GeneratedDecompositionStep(BaseModel):
    """Single-hop sub-instruction and its expected output from the LLM."""

    instruction: str
    answer: str


class InstructionGenerationResponse(BaseModel):
    """LLM output: multi-hop instruction, decomposition, informative paragraphs, final answer."""

    instruction: str
    decomposition: list[GeneratedDecompositionStep]
    informative_paragraphs: list[InformativeParagraph]
    answer: str


class DistractorParagraph(BaseModel):
    """Neighboring-concept distractor paragraph."""

    title: str
    text: str


class DistractorResponse(BaseModel):
    """LLM output: list of distractor paragraphs."""

    paragraphs: list[DistractorParagraph]


class RationaleResponse(BaseModel):
    """LLM output: rationale identifying informative vs distracting chunks."""

    rationale: str


class GroundedResponse(BaseModel):
    """LLM output: response grounded only in informative paragraphs."""

    response: str
