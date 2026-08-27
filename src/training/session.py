"""Shared setup for every Tinker entry point: loading the API key, resolving which renderer to
use for a model, naming the run for the metrics dashboard, applying command line overrides, and
locating the latest checkpoint of a previous run.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import chz
from chz.blueprint import EntrypointHelpException
from dotenv import load_dotenv

T = TypeVar("T")


def load_api_key() -> None:
    """Load TINKER_API_KEY from the environment or .env.

    Raises:
        RuntimeError: If the key is not set, since every Tinker call needs it.
    """
    load_dotenv()
    if not os.getenv("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY is not set. Add it to .env or export it. "
            "Create a key at https://tinker-console.thinkingmachines.ai"
        )


def run_name(stage: str) -> str:
    """Build the name this run reports to Weights & Biases.

    Every hyperparameter is uploaded separately as the run config, so the name
    only has to distinguish one run from the next.

    Params:
        stage: Training stage, "sft" or "rl".

    Returns:
        The run name, stamped with the local start time.
    """
    return f"{stage}-{datetime.now():%Y%m%d-%H%M%S}"


def resolve_renderer_name(model_name: str, renderer_name: str | None) -> str:
    """Pick the renderer that converts chat messages to tokens.

    Params:
        model_name: The base model being trained.
        renderer_name: Explicit renderer, or None to use the model's default.

    Returns:
        The renderer name.
    """
    from tinker_cookbook import model_info

    if renderer_name is None:
        return model_info.get_recommended_renderer_name(model_name)
    model_info.warn_if_renderer_not_recommended(model_name, renderer_name)
    return renderer_name


def latest_checkpoint_path(log_path: Path, key: str = "state_path") -> str | None:
    """Find the most recent checkpoint written by a previous run.

    Params:
        log_path: Log directory of that run.
        key: "state_path" to resume or branch training, "sampler_path" to sample
            or export weights.

    Returns:
        The tinker:// path, or None if the run has no such checkpoint.
    """
    from tinker_cookbook import checkpoint_utils

    if not (log_path / "checkpoints.jsonl").exists():
        return None
    record = checkpoint_utils.get_last_checkpoint(str(log_path), required_key=key)
    return getattr(record, key) if record else None


def config_from_argv(blueprint: chz.Blueprint[T], argv: list[str]) -> T:
    """Apply command line overrides to a config blueprint.

    Params:
        blueprint: The blueprint to finalize.
        argv: Arguments in chz "field=value" form.

    Returns:
        The built config. Exits after printing help when --help is passed.
    """
    try:
        return blueprint.make_from_argv(argv)
    except EntrypointHelpException as help_text:
        print(help_text)
        raise SystemExit(0) from None
