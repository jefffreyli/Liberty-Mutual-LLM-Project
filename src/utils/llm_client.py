"""OpenAI client wrapper for structured LLM generation."""

import os
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from ..config import DEFAULT_MODEL
from ..data.token_tracker import TokenTracker

load_dotenv()

class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.token_tracker = TokenTracker(model=model)

    def generate(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        """Send prompt to the model and parse the response into the given Pydantic schema."""
        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
        )
        self.token_tracker.update(input_tokens=completion.usage.prompt_tokens, output_tokens=completion.usage.completion_tokens)
        response = completion.choices[0].message.parsed
        return response
    
    def get_total_cost(self) -> tuple[float, float]:
        """Return the total cost of the API calls."""
        return self.token_tracker.get_total_cost()


_global_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return a process-wide shared LLMClient instance."""
    global _global_llm_client
    if _global_llm_client is None:
        _global_llm_client = LLMClient()
    return _global_llm_client
