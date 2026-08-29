"""Routes a model name to its provider's client, giving each worker thread its own client and
token tracker so parallel calls can report a race free total cost. Callers name a model and get
back something that satisfies `LLMClient`; which provider serves it is a fact of the model table.
"""

import threading
from typing import Protocol

from dotenv import load_dotenv
from pydantic import BaseModel

from src.config.models import DEFAULT_MODEL, MODELS
from src.llm.anthropic_client import AnthropicClient
from src.llm.openai_client import OpenAIClient

load_dotenv()

# Provider name in src/config/models.py to the class that serves it.
PROVIDER_CLIENTS = {"openai": OpenAIClient, "anthropic": AnthropicClient}


class LLMClient(Protocol):
    """What every provider client offers, whichever family serves the model."""

    model: str

    def generate(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        """Prompt the model and parse the reply into a schema."""
        ...

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float | None,
    ) -> str:
        """Prompt the model for free-form text."""
        ...

    def get_total_cost(self) -> tuple[float, float]:
        """Report this client's spend as input and output dollars."""
        ...


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

    Raises:
        KeyError: If the model has no entry in src/config/models.py, since
            without one it can be neither routed nor priced.
    """
    if model not in MODELS:
        raise KeyError(
            f"No entry for model '{model}'. Add it to MODELS in src/config/models.py"
        )
    clients: dict[str, LLMClient] = _thread_local.__dict__.setdefault("clients", {})
    if model not in clients:
        client = PROVIDER_CLIENTS[MODELS[model]["provider"]](model=model)
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
