"""Supervised fine-tunes the base model on the generated multi-hop data with Tinker.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import chz
import datasets
import tinker
from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

from src.config import training as cfg
from src.training.data import (
    load_rows,
    read_conversations,
    split_rows,
    split_sizes,
    write_conversations,
)
from src.training.session import (
    config_from_argv,
    load_api_key,
    resolve_renderer_name,
    run_name,
)


@chz.chz
class SplitFileBuilder(ChatDatasetBuilder):
    """Builds the SFT datasets from one conversation file per split.

    Reading the splits from separate files rather than carving a slice off a
    combined file is what keeps the validation rows identical to the ones
    src/training/data.py assigns, so SFT and RL monitor the same rows.
    """

    train_path: str
    val_path: str

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Load the training and validation conversations.

        Returns:
            The training dataset and, when the validation file has rows, a
            dataset holding all of them in a single batch.
        """
        train_on_what = self.common_config.train_on_what or TrainOnWhat.ALL_ASSISTANT_MESSAGES

        def to_datum(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], self.renderer, self.common_config.max_length, train_on_what
            )

        train_rows = read_conversations(Path(self.train_path))
        val_rows = read_conversations(Path(self.val_path))
        train_dataset = SupervisedDatasetFromHFDataset(
            datasets.Dataset.from_list(train_rows),
            batch_size=self.common_config.batch_size,
            map_fn=to_datum,
        )
        val_dataset = (
            SupervisedDatasetFromHFDataset(
                datasets.Dataset.from_list(val_rows),
                batch_size=len(val_rows),
                map_fn=to_datum,
            )
            if val_rows
            else None
        )
        return train_dataset, val_dataset


def ensure_conversations() -> None:
    """Write one conversation file per split from the run JSON if they are missing.

    The test file is written for inspection only. No trainer reads it.
    """
    if cfg.SFT_TRAIN_PATH.exists() and cfg.SFT_VAL_PATH.exists():
        return
    rows = load_rows(cfg.DATA_PATH)
    test_size, val_size = split_sizes(len(rows), cfg.TEST_FRACTION, cfg.VAL_FRACTION)
    train_rows, val_rows, test_rows = split_rows(rows, test_size, val_size, cfg.SEED)
    for split_rows_, path in (
        (train_rows, cfg.SFT_TRAIN_PATH),
        (val_rows, cfg.SFT_VAL_PATH),
        (test_rows, cfg.SFT_TEST_PATH),
    ):
        write_conversations(split_rows_, path)
        print(f"Wrote {len(split_rows_)} conversations to {path}")


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """Assemble the SFT config from src/config/training.py.

    Returns:
        A blueprint whose fields can still be overridden from argv.
    """
    renderer_name = resolve_renderer_name(cfg.MODEL_NAME, cfg.RENDERER_NAME)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.MODEL_NAME,
        renderer_name=renderer_name,
        max_length=cfg.SFT_MAX_LENGTH,
        batch_size=cfg.SFT_BATCH_SIZE,
        # The prompt is fixed and only the answer is learned, so loss is applied
        # to assistant tokens only.
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset_builder = SplitFileBuilder(
        common_config=common_config,
        train_path=str(cfg.SFT_TRAIN_PATH),
        val_path=str(cfg.SFT_VAL_PATH),
    )
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": str(cfg.SFT_LOG_PATH),
            "model_name": cfg.MODEL_NAME,
            "recipe_name": "multihop_sft",
            "wandb_project": cfg.WANDB_PROJECT,
            "wandb_name": run_name("sft"),
            "renderer_name": renderer_name,
            "dataset_builder": dataset_builder,
            "learning_rate": cfg.SFT_LEARNING_RATE,
            "lr_schedule": cfg.SFT_LR_SCHEDULE,
            "num_epochs": cfg.SFT_NUM_EPOCHS,
            "lora_rank": cfg.LORA_RANK,
            "save_every": cfg.SFT_SAVE_EVERY,
            "eval_every": cfg.SFT_EVAL_EVERY,
        }
    )


def run() -> None:
    """Start supervised fine-tuning, applying any command line overrides to the config."""
    load_api_key()
    ensure_conversations()
    config = config_from_argv(build_config_blueprint(), sys.argv[1:])
    print(f"SFT: {config.model_name} via {config.renderer_name} -> {config.log_path}")
    # Resume rather than clobber, so an interrupted run continues where it stopped.
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))

