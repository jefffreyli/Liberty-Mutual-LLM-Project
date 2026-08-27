"""Schemas for seed examples pulled from an existing multi-hop dataset to inspire generation."""

from __future__ import annotations

from pydantic import BaseModel


class QuestionDecompositionStep(BaseModel):
    """A single-hop sub-question with its answer from the seed dataset."""
    question: str
    answer: str


class SeedExample(BaseModel):
    """A single seed example to inject into the instruction-generation prompt."""
    instruction: str
    paragraphs: str  # supporting paragraph text (may be empty for non-MuSiQue seeds)
    output: str
    question_decomposition: list[QuestionDecompositionStep] = []
