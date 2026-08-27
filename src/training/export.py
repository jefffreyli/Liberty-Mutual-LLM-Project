"""Exports a finished Tinker run by downloading its LoRA adapter, merging it into the base model
in HuggingFace format, and optionally pushing the result to the Hub.
"""

from __future__ import annotations

from pathlib import Path

from src.config import training as cfg
from src.training.session import latest_checkpoint_path



def export(run: str, repo_id: str | None, private: bool) -> None:
    """Download, merge, and optionally publish the latest weights of a run.

    Params:
        run: Which run to export, "sft" or "rl".
        repo_id: HuggingFace repo to push to, or None to only write locally.
        private: Whether a pushed repo is private.

    Raises:
        FileNotFoundError: If the run has no sampler checkpoint yet.
    """
    from tinker_cookbook import weights

    log_path = cfg.RUN_LOG_PATHS[run]
    tinker_path = latest_checkpoint_path(log_path, key="sampler_path")
    if tinker_path is None:
        raise FileNotFoundError(f"No sampler checkpoint in {log_path}; has the {run} run finished?")

    adapter_dir = Path(cfg.EXPORT_DIR) / run / "adapter"
    model_dir = Path(cfg.EXPORT_DIR) / run / "model"
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {tinker_path}")
    weights.download(tinker_path=tinker_path, output_dir=str(adapter_dir))

    print(f"Merging adapter into {cfg.MODEL_NAME}")
    weights.build_hf_model(
        base_model=cfg.MODEL_NAME,
        adapter_path=str(adapter_dir),
        output_path=str(model_dir),
        dtype="bfloat16",
    )
    print(f"Wrote merged model to {model_dir}")

    if repo_id:
        url = weights.publish_to_hf_hub(model_path=str(model_dir), repo_id=repo_id, private=private)
        print(f"Pushed to {url}")
