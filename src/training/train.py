"""Supervised fine-tune an open-source model on a SAIL run JSON.

Loads the run JSON specified in `src/training/config.py`, converts rows to
ChatML messages, and trains with TRL's SFTTrainer + PEFT LoRA adapters.

Run:
    python -m src.training.train
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset as HFDataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.data.dataset import Dataset
from src.schema import TrainingRow
import src.training.config as cfg


def load_chatml_examples(data_path: Path) -> list[dict]:
    """Load a run JSON and return ChatML-formatted training examples.
    """
    with open(data_path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "data" in payload:
        raw_rows = payload["data"]
    else:
        raise ValueError(
            f"Unrecognized run JSON shape at {data_path}: "
            f"expected dict with 'data' key."
        )

    temp = Dataset(num_rows=0)
    temp.rows = [TrainingRow.model_validate(r) for r in raw_rows]
    return temp.convert_to_chatml_format()


def build_hf_dataset(data_path: Path, eval_fraction: float, seed: int) -> tuple[HFDataset, HFDataset | None]:
    """Build train (and optional eval) HuggingFace datasets from a run JSON."""
    examples = load_chatml_examples(data_path)
    print(f"Loaded {len(examples)} ChatML examples from {data_path}")

    hf_dataset = HFDataset.from_list(examples)
    hf_dataset = hf_dataset.remove_columns(
        [c for c in hf_dataset.column_names if c != "messages"]
    )

    if eval_fraction <= 0 or len(hf_dataset) < 10:
        return hf_dataset, None

    split = hf_dataset.train_test_split(test_size=eval_fraction, seed=seed)
    return split["train"], split["test"]


def main() -> None:
    if not cfg.DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {cfg.DATA_PATH}")

    train_ds, eval_ds = build_hf_dataset(cfg.DATA_PATH, cfg.EVAL_FRACTION, cfg.SEED)
    print(
        f"train={len(train_ds)} examples"
        + (f", eval={len(eval_ds)} examples" if eval_ds is not None else "")
    )

    # SFTTrainer auto-applies the chat template, but we need a tokenizer with a
    # pad token (Qwen models default to <|endoftext|>) so collators don't crash.
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = (
        LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        if cfg.USE_LORA
        else None
    )

    sft_config = SFTConfig(
        output_dir=str(cfg.OUTPUT_DIR),
        num_train_epochs=cfg.NUM_EPOCHS,
        per_device_train_batch_size=cfg.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=cfg.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=cfg.LEARNING_RATE,
        max_length=cfg.MAX_LENGTH,
        assistant_only_loss=True,
        packing=False,
        bf16=cfg.BF16,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=cfg.SEED,
    )

    trainer = SFTTrainer(
        model=cfg.MODEL_NAME,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(cfg.OUTPUT_DIR))
    print(f"Saved final model/adapter to {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
