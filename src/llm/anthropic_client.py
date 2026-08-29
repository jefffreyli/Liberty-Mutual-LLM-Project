"""Anthropic-backed client, exposing the same structured generation and free-form completion
surface as the OpenAI client so a judge or an evaluation baseline can name either family's model
without the caller knowing which one it got.
"""

import os

import anthropic
from pydantic import BaseModel

from src.config.models import STRUCTURED_MAX_TOKENS
from src.llm.tokens import TokenTracker


class AnthropicClient:
    """One Anthropic connection and its token tracker."""

    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.token_tracker = TokenTracker(model=model)

    def generate(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        """Prompt the model and parse the reply into a schema.

        Params:
            prompt: The prompt text.
            response_model: Pydantic schema the reply must satisfy.

        Returns:
            The parsed reply.

        Raises:
            ValueError: If the model returns a reply that cannot be parsed.
        """
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=STRUCTURED_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            output_format=response_model,
        )
        self._track(response)
        if response.parsed_output is None:
            raise ValueError(
                f"LLM returned unparseable response for schema {response_model.__name__}. "
                f"Stop reason: {response.stop_reason}"
            )
        return response.parsed_output

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float | None,
    ) -> str:
        """Prompt the model for free-form text.

        Params:
            system: The system turn, which this API takes beside the messages
                rather than inside them.
            messages: Alternating user and assistant turns, ending on a user turn.
            max_tokens: Generation budget, shared with thinking tokens on a
                thinking model.
            temperature: Sampling temperature, or None to leave it unset. The
                current reasoning models reject the parameter with a 400 rather
                than ignoring it, so it cannot be defaulted here.

        Returns:
            The reply text, concatenated across text blocks and empty if the
            model returned none.
        """
        options = {} if temperature is None else {"temperature": temperature}
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            **options,
        )
        self._track(response)
        return "".join(block.text for block in response.content if block.type == "text")

    def _track(self, response) -> None:
        """Record one call's billed tokens.

        Params:
            response: The message the API returned.
        """
        self.token_tracker.update(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def get_total_cost(self) -> tuple[float, float]:
        """Report this client's spend.

        Returns:
            Input and output cost in dollars.
        """
        return self.token_tracker.get_total_cost()
