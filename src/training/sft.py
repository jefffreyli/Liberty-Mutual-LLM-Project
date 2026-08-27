"""Supervised fine-tunes the base model on the generated multi-hop data with Tinker, teaching
the answer format defined in src/training/format.py. The conversation JSONL is built on demand,
and any config field can be overridden on the command line.
"""

from __future__ import annotations

import asyncio
import sys

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from src.config import training as cfg
from src.training.data import load_rows, split_rows, split_sizes, write_conversations
from src.training.session import (
    config_from_argv,
    load_api_key,
    resolve_renderer_name,
    run_name,
)


def ensure_conversations() -> None:
    """Build the SFT conversation JSONL from the run JSON if it is missing.

    Only the training split is written, so the rows src/evaluation scores stay unseen.
    """
    if cfg.SFT_JSONL_PATH.exists():
        return
    rows = load_rows(cfg.DATA_PATH)
    test_size, val_size = split_sizes(len(rows), cfg.TEST_FRACTION, cfg.VAL_FRACTION)
    train_rows, val_rows, _ = split_rows(rows, test_size, val_size, cfg.SEED)
    # The builder carves its own held out slice off the front of this file, so the
    # validation rows ride along with the training rows rather than in their own file.
    write_conversations(train_rows + val_rows, cfg.SFT_JSONL_PATH)
    print(
        f"Wrote {len(train_rows) + len(val_rows)} conversations to {cfg.SFT_JSONL_PATH} "
        f"({len(train_rows)} train + {len(val_rows)} validation)"
    )


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    """Assemble the SFT config from src/config/training.py.

    Returns:
        A blueprint whose fields can still be overridden from argv.
    """
    _, val_size = split_sizes(
        len(load_rows(cfg.DATA_PATH)), cfg.TEST_FRACTION, cfg.VAL_FRACTION
    )
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
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(cfg.SFT_JSONL_PATH),
        # The validation rows, used for held out NLL during the run.
        test_size=val_size,
        shuffle_seed=cfg.SEED,
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

