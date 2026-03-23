"""Public schema exports for the SAIL pipeline."""

from .core import DecompositionStep, SearchResult, TrainingRow
from .generation import (
    DistractorParagraph,
    DistractorResponse,
    GeneratedDecompositionStep,
    GroundedResponse,
    InformativeParagraph,
    InstructionGenerationResponse,
    RationaleResponse,
)
from .evaluation import EvaluationResult, MetricScore, RubricVerdict

# Backward compatibility alias for older imports/usages.
QuestionGenerationResponse = InstructionGenerationResponse

__all__ = [
    "DecompositionStep",
    "SearchResult",
    "TrainingRow",
    "DistractorParagraph",
    "DistractorResponse",
    "GeneratedDecompositionStep",
    "GroundedResponse",
    "InformativeParagraph",
    "InstructionGenerationResponse",
    "QuestionGenerationResponse",
    "RationaleResponse",
    "EvaluationResult",
    "MetricScore",
    "RubricVerdict",
]
