"""OpenAI-backed client: structured generation for the teacher and the judges, and free-form
completion for the API baselines that src/evaluation samples answers from.
"""

import os

from openai import OpenAI
from pydantic import BaseModel

from src.llm.tokens import TokenTracker


class OpenAIClient:
    """One OpenAI connection and its token tracker."""

    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
        )
        self.token_tracker.update(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )
        response = completion.choices[0].message.parsed
        if response is None:
            raise ValueError(
                f"LLM returned unparseable response for schema {response_model.__name__}. "
                f"Raw content: {completion.choices[0].message.content!r}"
            )
        return response

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float | None,
    ) -> str:
        """Prompt the model for free-form text.

        Params:
            system: The system turn.
            messages: Alternating user and assistant turns, ending on a user turn.
            max_tokens: Generation budget.
            temperature: Sampling temperature, or None to leave it unset for the
                reasoning models that reject the parameter.

        Returns:
            The reply text, empty if the model returned no content.
        """
        options = {} if temperature is None else {"temperature": temperature}
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            max_completion_tokens=max_tokens,
            **options,
        )
        self.token_tracker.update(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )
        return completion.choices[0].message.content or ""

    def get_total_cost(self) -> tuple[float, float]:
        """Report this client's spend.

        Returns:
            Input and output cost in dollars.
        """
        return self.token_tracker.get_total_cost()
