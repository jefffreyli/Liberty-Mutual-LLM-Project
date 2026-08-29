"""Samples one answer per held out row from whatever a baseline names: a set of Tinker weights or
a hosted API model. Both backends send the same system prompt, the same exemplars, and the same
rendered search pool, so the only thing that differs between two runs is the model itself.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from src.config import training as train_cfg
from src.config.models import SAMPLE_WORKERS
from src.evaluation.baselines import Baseline
from src.evaluation.fewshot import build_messages, select_exemplars
from src.llm.client import get_llm_client
from src.schema import TrainingRow
from src.training.format import SYSTEM_PROMPT
from src.training.session import (
    checkpoint_path,
    latest_checkpoint_path,
    resolve_renderer_name,
)


class Sampler(Protocol):
    """Generates one answer per row from a single model."""

    def sample(self, rows: list[TrainingRow]) -> list[str]:
        """Generate an answer for every row, in row order."""
        ...


def resolve_model_path(run: str, checkpoint: str | None = None) -> str | None:
    """Find the weights to evaluate for a named run.

    Params:
        run: "base" for the untrained model, otherwise a key of train_cfg.RUN_LOG_PATHS.
        checkpoint: Name of one checkpoint to score, such as "000200", or None
            for the run's latest.

    Returns:
        The tinker:// sampler path, or None to use base weights.

    Raises:
        FileNotFoundError: If the run has no matching sampler checkpoint.
    """
    if run == "base":
        return None
    log_path = train_cfg.RUN_LOG_PATHS[run]
    if checkpoint is None:
        model_path = latest_checkpoint_path(log_path, key="sampler_path")
    else:
        model_path = checkpoint_path(log_path, checkpoint, key="sampler_path")
    if model_path is None:
        target = checkpoint or "latest"
        raise FileNotFoundError(f"No sampler checkpoint {target!r} for the {run} run")
    return model_path


class TinkerSampler:
    """Samples from base weights or a checkpoint of a finished Tinker run."""

    def __init__(
        self,
        model_path: str | None,
        model_name: str,
        renderer_name: str | None,
        exemplars: list[TrainingRow],
        max_tokens: int,
        temperature: float | None,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.exemplars = exemplars
        self.max_tokens = max_tokens
        # Tinker always samples at some temperature, so a baseline that leaves
        # it unset for an API model still samples greedily here.
        self.temperature = 0.0 if temperature is None else temperature

    def sample(self, rows: list[TrainingRow]) -> list[str]:
        """Generate an answer for every row.

        Params:
            rows: Rows to answer.

        Returns:
            Raw completions in row order.
        """
        return asyncio.run(self._sample_async(rows))

    async def _sample_async(self, rows: list[TrainingRow]) -> list[str]:
        """Run the rollouts concurrently.

        Params:
            rows: Rows to answer.

        Returns:
            Raw completions in row order.
        """
        import tinker
        from tinker_cookbook import renderers
        from tinker_cookbook.completers import TinkerMessageCompleter
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        renderer_name = resolve_renderer_name(self.model_name, self.renderer_name)
        renderer = renderers.get_renderer(
            renderer_name, tokenizer=get_tokenizer(self.model_name)
        )
        sampling_client = tinker.ServiceClient().create_sampling_client(
            model_path=self.model_path,
            base_model=None if self.model_path else self.model_name,
        )
        completer = TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=self.max_tokens,
            stop_condition=renderer.get_stop_sequences(),
            temperature=self.temperature,
        )
        limit = asyncio.Semaphore(SAMPLE_WORKERS)

        async def answer(row: TrainingRow) -> str:
            async with limit:
                message = await completer(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *build_messages(row, self.exemplars),
                    ]
                )
            return renderers.get_text_content(message)

        return await asyncio.gather(*[answer(row) for row in rows])


class APISampler:
    """Samples from a hosted model through the provider clients in src/llm."""

    def __init__(
        self,
        model: str,
        exemplars: list[TrainingRow],
        max_tokens: int,
        temperature: float | None,
    ):
        self.model = model
        self.exemplars = exemplars
        self.max_tokens = max_tokens
        self.temperature = temperature

    def sample(self, rows: list[TrainingRow], workers: int = SAMPLE_WORKERS) -> list[str]:
        """Generate an answer for every row.

        A failed call yields an empty completion rather than raising, which
        parses as malformed and scores zero, so one bad call costs a row instead
        of the whole run.

        Params:
            rows: Rows to answer.
            workers: Parallel API calls, each on its own thread-local client.

        Returns:
            Raw completions in row order.
        """

        def answer(row: TrainingRow) -> str:
            try:
                return get_llm_client(self.model).complete(
                    system=SYSTEM_PROMPT,
                    messages=build_messages(row, self.exemplars),
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except Exception as error:
                print(f"  sampling failed on row {row.id}: {error}")
                return ""

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(answer, rows))


def build_sampler(baseline: Baseline, checkpoint: str | None = None) -> Sampler:
    """Build the sampler a baseline calls for.

    Params:
        baseline: The baseline to score.
        checkpoint: Name of one checkpoint to score for a Tinker baseline, or
            None for the run's latest. Ignored by API baselines.

    Returns:
        A sampler over the baseline's model.
    """
    exemplars = select_exemplars(baseline.few_shot)
    if baseline.kind == "tinker":
        return TinkerSampler(
            model_path=resolve_model_path(baseline.target, checkpoint),
            model_name=baseline.base_model or train_cfg.MODEL_NAME,
            renderer_name=baseline.renderer or train_cfg.RENDERER_NAME,
            exemplars=exemplars,
            max_tokens=baseline.max_tokens,
            temperature=baseline.temperature,
        )
    return APISampler(
        model=baseline.target,
        exemplars=exemplars,
        max_tokens=baseline.max_tokens,
        temperature=baseline.temperature,
    )
