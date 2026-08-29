"""Samples answers from a model under evaluation, either the base weights or a Tinker checkpoint
from a finished run, using the same prompt format the model was trained on.
"""

import asyncio

from src.config.models import SAMPLE_TEMPERATURE, SAMPLE_WORKERS
from src.schema import TrainingRow
from src.config import training as train_cfg
from src.training.format import SYSTEM_PROMPT, build_user_prompt
from src.training.session import (
    checkpoint_path,
    latest_checkpoint_path,
    resolve_renderer_name,
)



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


class ResponseSampler:
    """Generates one answer per row from a single set of weights."""

    def __init__(
        self,
        model_path: str | None,
        max_tokens: int = train_cfg.RL_MAX_TOKENS,
        temperature: float = SAMPLE_TEMPERATURE,
    ):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature

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

        renderer_name = resolve_renderer_name(train_cfg.MODEL_NAME, train_cfg.RENDERER_NAME)
        renderer = renderers.get_renderer(
            renderer_name, tokenizer=get_tokenizer(train_cfg.MODEL_NAME)
        )
        sampling_client = tinker.ServiceClient().create_sampling_client(
            model_path=self.model_path,
            base_model=None if self.model_path else train_cfg.MODEL_NAME,
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
                        {"role": "user", "content": build_user_prompt(row)},
                    ]
                )
            return renderers.get_text_content(message)

        return await asyncio.gather(*[answer(row) for row in rows])
