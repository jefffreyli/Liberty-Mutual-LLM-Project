"""Entry point for training data prep: converts a generated run JSON into the conversation JSONL
that Tinker's SFT loop reads, prints an example, and reports the token length distribution so the
max sequence length can be set before training.

Run:
    python3 -m scripts.prepare_data
"""

from src.config import training as cfg
from src.training.data import (
    load_rows,
    report_token_lengths,
    split_rows,
    split_sizes,
    write_conversations,
)
from src.training.format import row_to_messages


def main() -> None:
    rows = load_rows(cfg.DATA_PATH)
    test_size, val_size = split_sizes(len(rows), cfg.TEST_FRACTION, cfg.VAL_FRACTION)
    train_rows, val_rows, test_rows = split_rows(rows, test_size, val_size, cfg.SEED)
    print(f"Loaded {len(rows)} rows from {cfg.DATA_PATH}")

    write_conversations(train_rows + val_rows, cfg.SFT_JSONL_PATH)
    print(
        f"Split {len(rows)} rows into {len(train_rows)} train / {len(val_rows)} validation / "
        f"{len(test_rows)} test"
    )
    print(f"Wrote {len(train_rows) + len(val_rows)} conversations to {cfg.SFT_JSONL_PATH}")

    print("\nExample conversation:")
    for message in row_to_messages(train_rows[0]):
        print(f"--- {message['role']} ---\n{message['content'][:600]}\n")

    report_token_lengths(train_rows, cfg.MODEL_NAME, cfg.SFT_MAX_LENGTH)


if __name__ == "__main__":
    main()
