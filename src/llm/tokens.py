"""Accumulates the token usage reported by one LLM client and converts it into dollars using
the per-model prices in the model config.
"""

from src.config.models import MODELS

TOKENS_PER_PRICE_UNIT = 1_000_000


class TokenTracker:
    """Running token totals and cost for a single model."""

    def __init__(self, model: str):
        if model not in MODELS:
            raise KeyError(
                f"No price for model '{model}'. Add it to MODELS in src/config/models.py"
            )
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

    def update(self, input_tokens: int, output_tokens: int) -> None:
        """Add the usage reported by one API call.

        Params:
            input_tokens: Prompt tokens billed.
            output_tokens: Completion tokens billed.
        """
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def get_total_cost(self) -> tuple[float, float]:
        """Convert the accumulated tokens into dollars.

        Returns:
            Input and output cost so far.
        """
        price = MODELS[self.model]
        return (
            self.input_tokens / TOKENS_PER_PRICE_UNIT * price["input"],
            self.output_tokens / TOKENS_PER_PRICE_UNIT * price["output"],
        )
