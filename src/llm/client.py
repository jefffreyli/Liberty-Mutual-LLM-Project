"""Wraps the OpenAI client for structured generation, giving each worker thread its own client
and token tracker so parallel row generation can report a race free total cost.
"""

import os
import threading

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.config.models import DEFAULT_MODEL
from src.llm.tokens import TokenTracker

load_dotenv()


class LLMClient:
    """One OpenAI connection and its token tracker."""

    def __init__(self, model: str = DEFAULT_MODEL):
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

    def get_total_cost(self) -> tuple[float, float]:
        """Report this client's spend.

        Returns:
            Input and output cost in dollars.
        """
        return self.token_tracker.get_total_cost()


# Each thread gets its own client per model so token tracking is race free and
# the underlying connection pool is not shared.
_thread_local = threading.local()
_all_clients: list[LLMClient] = []
_clients_lock = threading.Lock()


def get_llm_client(model: str = DEFAULT_MODEL) -> LLMClient:
    """Return this thread's client for a model, creating it on first use.

    Params:
        model: Model the caller wants, defaulting to the generation model.

    Returns:
        The thread-local client for that model.
    """
    clients: dict[str, LLMClient] = _thread_local.__dict__.setdefault("clients", {})
    if model not in clients:
        client = LLMClient(model=model)
        clients[model] = client
        with _clients_lock:
            _all_clients.append(client)
    return clients[model]


def get_aggregate_cost() -> tuple[float, float]:
    """Sum the spend of every client created so far.

    Returns:
        Input and output cost in dollars across all threads.
    """
    total_input, total_output = 0.0, 0.0
    with _clients_lock:
        for client in _all_clients:
            client_input, client_output = client.get_total_cost()
            total_input += client_input
            total_output += client_output
    return total_input, total_output
