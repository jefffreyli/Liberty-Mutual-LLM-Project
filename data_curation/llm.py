"""OpenAI client wrapper for structured LLM generation."""

import os
from typing import Optional, TypeVar

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gpt-4o"

_client: Optional[OpenAI] = None

T = TypeVar("T", bound=BaseModel)


def get_client() -> OpenAI:
    """Return a singleton OpenAI client (uses OPENAI_API_KEY from env)."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def generate(prompt: str, response_model: type[T], model: str = DEFAULT_MODEL) -> T:
    """Send prompt to the model and parse the response into the given Pydantic schema."""
    client = get_client()
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=response_model,
    )
    return completion.choices[0].message.parsed
