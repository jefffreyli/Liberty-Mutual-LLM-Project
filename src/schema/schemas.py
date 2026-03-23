"""Backward-compatible schema re-exports."""

from . import (
    DecompositionStep,
    DistractorParagraph,
    DistractorResponse,
    EvaluationResult,
    GeneratedDecompositionStep,
    GroundedResponse,
    InformativeParagraph,
    InstructionGenerationResponse,
    MetricScore,
    RationaleResponse,
    RubricVerdict,
    SearchResult,
    TrainingRow,
)

# Backward compatibility alias for older imports/usages.
QuestionGenerationResponse = InstructionGenerationResponse

__all__ = [
    "DecompositionStep",
    "DistractorParagraph",
    "DistractorResponse",
    "EvaluationResult",
    "GeneratedDecompositionStep",
    "GroundedResponse",
    "InformativeParagraph",
    "InstructionGenerationResponse",
    "QuestionGenerationResponse",
    "MetricScore",
    "RationaleResponse",
    "RubricVerdict",
    "SearchResult",
    "TrainingRow",
]
