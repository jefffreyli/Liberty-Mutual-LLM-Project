"""OpenAI client wrapper for structured LLM generation."""

import os
import threading

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
        self.token_tracker.update(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )
        response = completion.choices[0].message.parsed
        if response is None:
            raise ValueError(
                f"LLM returned unparseable response for schema "
                f"{response_model.__name__}. Raw content: "
                f"{completion.choices[0].message.content!r}"
            )
        return response

    def get_total_cost(self) -> tuple[float, float]:
        """Return the total cost of the API calls."""
        return self.token_tracker.get_total_cost()


# Thread-local storage: each thread gets its own LLMClient so token tracking
# is race-condition-free and the underlying OpenAI connection pool is not shared.
_thread_local = threading.local()
_all_clients: list[LLMClient] = []
_clients_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """Return a thread-local LLMClient instance, creating one on first access."""
    if not hasattr(_thread_local, "client"):
        client = LLMClient()
        _thread_local.client = client
        with _clients_lock:
            _all_clients.append(client)
    return _thread_local.client


def get_aggregate_cost() -> tuple[float, float]:
    """Sum input/output costs across all thread-local clients."""
    total_input, total_output = 0.0, 0.0
    with _clients_lock:
        for client in _all_clients:
            inp, out = client.get_total_cost()
            total_input += inp
            total_output += out
    return total_input, total_output
