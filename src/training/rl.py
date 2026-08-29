"""Reinforcement learns on top of the SFT checkpoint with Tinker's GRPO loop, where each
environment shows one instruction plus its noisy search pool and scores the sampled answer with
the programmatic reward in src/training/metrics.py.
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, renderers
from tinker_cookbook.rl import train
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.schema import TrainingRow
from src.config import training as cfg
from src.training.data import load_rows, split_rows, split_sizes
from src.training.format import SYSTEM_PROMPT, build_user_prompt, gold_informative_ids
from src.training.metrics import Grade, grade_completion
from src.training.session import (
    config_from_argv,
    latest_checkpoint_path,
    load_api_key,
    resolve_renderer_name,
    run_name,
)

# Tags split the logged reward by row type. Unanswerable rows are the ones whose
# group can collapse to zero advantage once the model reliably abstains, which is
# invisible in a single blended reward curve.
ANSWERABLE_TAG = "answerable"
UNANSWERABLE_TAG = "unanswerable"


def row_tag(row: TrainingRow) -> str:
    """Name the row type this row belongs to, for per type metrics.

    Params:
        row: The training row.

    Returns:
        The logging tag.
    """
    return ANSWERABLE_TAG if row.is_answerable else UNANSWERABLE_TAG


class MultiHopEnv(ProblemEnv):
    """A single turn environment that scores one answer to one multi-hop instruction."""

    def __init__(
        self,
        row: TrainingRow,
        renderer: renderers.Renderer,
        citation_weight: float,
        answer_weight: float,
        leakage_weight: float,
        format_coef: float,
    ):
        super().__init__(
            renderer,
            convo_prefix=[{"role": "system", "content": SYSTEM_PROMPT}],
            format_coef=format_coef,
        )
        self.row = row
        self.weights = {
            "citation_weight": citation_weight,
            "answer_weight": answer_weight,
            "leakage_weight": leakage_weight,
        }
        self._grades: dict[str, Grade] = {}

    def _grade(self, sample_str: str) -> Grade:
        """Grade a sampled answer, memoized because the base class scores it twice.

        Params:
            sample_str: The decoded model answer.

        Returns:
            The grade for this answer.
        """
        if sample_str not in self._grades:
            self._grades[sample_str] = grade_completion(sample_str, self.row, **self.weights)
        return self._grades[sample_str]

    def get_question(self) -> str:
        """Return the user turn for this row.

        Returns:
            The instruction and its rendered search pool.
        """
        return build_user_prompt(self.row)

    def check_format(self, sample_str: str) -> bool:
        """Check that the answer can be parsed into cited IDs and a response.

        Params:
            sample_str: The decoded model answer.

        Returns:
            True if the answer is well formed.
        """
        return self._grade(sample_str).format_ok

    def check_answer(self, sample_str: str) -> float:
        """Score answer quality.

        Params:
            sample_str: The decoded model answer.

        Returns:
            A graded score in [0, 1] rather than a bool, which the base class
            adds to the reward as is.
        """
        return self._grade(sample_str).score

    def get_reference_answer(self) -> str:
        """Return the target used for rollout logging.

        Returns:
            The gold informative IDs and the teacher response.
        """
        return f"Informative IDs: {gold_informative_ids(self.row)}\n{self.row.response}"

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Score the sampled answer and attach the reward components as metrics.

        Params:
            action: Token IDs of the model's answer.
            extra: Optional action metadata, forwarded to the base class.

        Returns:
            The step result, with per component metrics for monitoring.
        """
        result = await super().step(action, extra=extra)
        message, _ = self.renderer.parse_response(action)
        result.metrics.update(self._grade(renderers.get_text_content(message)).as_metrics())
        return result


class MultiHopDataset(RLDataset):
    """Batches of training rows, one environment group per row."""

    def __init__(
        self,
        rows: list[TrainingRow],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        weights: dict[str, float],
        format_coef: float,
    ):
        self.rows = rows
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.weights = weights
        self.format_coef = format_coef

    def __len__(self) -> int:
        """Return the number of batches.

        Returns:
            Batch count, dropping no rows.
        """
        return math.ceil(len(self.rows) / self.batch_size)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Build the environment groups for one batch.

        Params:
            index: Batch index.

        Returns:
            One group builder per row in the batch.
        """
        start = index * self.batch_size
        batch = self.rows[start : start + self.batch_size]
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    MultiHopEnv,
                    row,
                    self.renderer,
                    format_coef=self.format_coef,
                    **self.weights,
                ),
                num_envs=self.group_size,
                dataset_name=row_tag(row),
            )
            for row in batch
        ]


@chz.chz
class MultiHopDatasetBuilder(RLDatasetBuilder):
    """Builds the train and held out RL datasets from a generated run JSON."""

    data_path: str
    model_name_for_tokenizer: str
    renderer_name: str
    batch_size: int
    group_size: int
    test_fraction: float = 0.05
    val_fraction: float = 0.05
    seed: int = 0
    citation_weight: float = 0.5
    answer_weight: float = 0.5
    leakage_weight: float = 0.2
    format_coef: float = 0.1

    async def __call__(self) -> tuple[MultiHopDataset, MultiHopDataset | None]:
        """Load and split the rows.

        Returns:
            The training dataset and, when there are validation rows, a dataset
            sampled once per row for evaluation during the run. The test rows are
            left for src/evaluation and are never sampled here.
        """
        rows = load_rows(Path(self.data_path))
        test_size, val_size = split_sizes(len(rows), self.test_fraction, self.val_fraction)
        train_rows, val_rows, _ = split_rows(rows, test_size, val_size, self.seed)

        renderer = renderers.get_renderer(
            self.renderer_name, tokenizer=get_tokenizer(self.model_name_for_tokenizer)
        )
        weights = {
            "citation_weight": self.citation_weight,
            "answer_weight": self.answer_weight,
            "leakage_weight": self.leakage_weight,
        }
        make = partial(
            MultiHopDataset, renderer=renderer, weights=weights, format_coef=self.format_coef
        )
        train_dataset = make(train_rows, batch_size=self.batch_size, group_size=self.group_size)
        # Evaluation needs one sample per row, not a group to center advantages over.
        val_dataset = (
            make(val_rows, batch_size=self.batch_size, group_size=1) if val_rows else None
        )
        return train_dataset, val_dataset


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """Assemble the RL config from src/config/training.py.

    Returns:
        A blueprint whose fields can still be overridden from argv.
    """
    renderer_name = resolve_renderer_name(cfg.MODEL_NAME, cfg.RENDERER_NAME)
    dataset_builder = MultiHopDatasetBuilder(
        data_path=str(cfg.DATA_PATH),
        model_name_for_tokenizer=cfg.MODEL_NAME,
        renderer_name=renderer_name,
        batch_size=cfg.RL_GROUPS_PER_BATCH,
        group_size=cfg.RL_GROUP_SIZE,
        test_fraction=cfg.TEST_FRACTION,
        val_fraction=cfg.VAL_FRACTION,
        seed=cfg.SEED,
        citation_weight=cfg.REWARD_CITATION_WEIGHT,
        answer_weight=cfg.REWARD_ANSWER_WEIGHT,
        leakage_weight=cfg.REWARD_LEAKAGE_WEIGHT,
        format_coef=cfg.REWARD_FORMAT_COEF,
    )
    # RL from the base model has to discover the answer format by chance, so we
    # start from the SFT weights when that run has a checkpoint.
    load_checkpoint_path = (
        latest_checkpoint_path(cfg.SFT_LOG_PATH) if cfg.RL_INIT_FROM_SFT else None
    )
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": str(cfg.RL_LOG_PATH),
            "model_name": cfg.MODEL_NAME,
            "recipe_name": "multihop_rl",
            "wandb_project": cfg.WANDB_PROJECT,
            "wandb_name": run_name("rl"),
            "renderer_name": renderer_name,
            "dataset_builder": dataset_builder,
            "load_checkpoint_path": load_checkpoint_path,
            "learning_rate": cfg.RL_LEARNING_RATE,
            "max_tokens": cfg.RL_MAX_TOKENS,
            "lora_rank": cfg.LORA_RANK,
            "kl_penalty_coef": cfg.RL_KL_PENALTY_COEF,
            "save_every": cfg.RL_SAVE_EVERY,
            "eval_every": cfg.RL_EVAL_EVERY,
        }
    )


def run() -> None:
    """Start the GRPO run, applying any command line overrides to the config."""
    load_api_key()
    config = config_from_argv(build_config_blueprint(), sys.argv[1:])
    start = config.load_checkpoint_path or f"{config.model_name} (base weights)"
    print(f"RL: starting from {start} -> {config.log_path}")
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))

