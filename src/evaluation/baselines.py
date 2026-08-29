"""The set of models an evaluation run can score, and what each one needs to be sampled fairly.

A baseline is either a set of Tinker weights (the base model or a checkpoint from a finished run)
or an API model prompted through src/llm. Adding a comparison model means adding an entry here
and its price in src/config/models.py, and nothing else.
"""

from dataclasses import dataclass
from typing import Literal

from src.config import training as train_cfg

# Exemplars shown to every prompted baseline. The fine-tuned checkpoints were
# taught the answer format by training on it, so comparing them against a model
# that has never seen the format measures format compliance rather than
# capability. Three is enough to establish the shape of a turn without pushing
# the pool out of the prompt.
DEFAULT_FEW_SHOT = 3

# Generation budget for the API baselines. Larger than the rollout budget the
# trainers use, because a thinking model spends the same budget on reasoning
# before it writes anything, and a truncated answer scores zero for length
# rather than for being wrong.
API_MAX_TOKENS = 16000


@dataclass(frozen=True)
class Baseline:
    """One model an evaluation run can score.

    Params:
        name: The key `scripts/evaluate.py --run` takes, and the results filename.
        kind: "tinker" for local weights, "api" for a hosted model.
        target: A key of train_cfg.RUN_LOG_PATHS or "base" for a Tinker
            baseline, otherwise the API model id.
        few_shot: Exemplars prepended to the prompt.
        temperature: Sampling temperature, or None to leave it unset. The
            reasoning models reject the parameter with a 400 rather than
            ignoring it, so it cannot be defaulted across baselines.
        max_tokens: Generation budget per answer.
        base_model: Weights a Tinker baseline samples from, or None for the
            model the trainers fine-tune. Set it to compare against a different
            size of the same family. Ignored by API baselines.
        renderer: Renderer for those weights, or None for the trained model's.
            A model whose default renderer leaves a thinking block open needs
            naming here, because the answer format expects the ID line first.
    """

    name: str
    kind: Literal["tinker", "api"]
    target: str
    few_shot: int = 0
    temperature: float | None = 0.0
    max_tokens: int = train_cfg.RL_MAX_TOKENS
    base_model: str | None = None
    renderer: str | None = None


_BASELINES = (
    # The ladder: what training bought, isolated from what prompting alone gets.
    Baseline("base", "tinker", "base"),
    Baseline("base-fewshot", "tinker", "base", few_shot=DEFAULT_FEW_SHOT),
    Baseline("sft", "tinker", "sft"),
    Baseline("rl", "tinker", "rl"),
    # The same family three times the size, prompted rather than trained: what
    # separates "the fine-tuning worked" from "you needed a bigger model". This
    # is the largest dense model Tinker will sample. Tinker serves no 3.5 dense
    # model above 9B, so it is three minor versions ahead as well as larger and
    # does not isolate scale on its own; it is the stronger opponent for that,
    # which makes beating it a conservative result. It is sampled through the
    # 3.5 renderer, which the cookbook's model table does not know this model
    # by: naming it here is required, because asking for its default renderer
    # raises a KeyError.
    Baseline(
        "qwen3.8-27b",
        "tinker",
        "base",
        few_shot=DEFAULT_FEW_SHOT,
        base_model="Qwen/Qwen3.8-27B",
        renderer=train_cfg.RENDERER_NAME,
    ),
    # The teacher that wrote the training data, and so the ceiling the
    # fine-tuned model is distilling toward.
    Baseline("gpt-4o", "api", "gpt-4o", DEFAULT_FEW_SHOT, 0.0, API_MAX_TOKENS),
    # Frontier ceiling, one model per family so the comparison is not graded
    # entirely within the judge's own family.
    Baseline("gpt-5.5", "api", "gpt-5.5", DEFAULT_FEW_SHOT, None, API_MAX_TOKENS),
    Baseline("claude-opus-5", "api", "claude-opus-5", DEFAULT_FEW_SHOT, None, API_MAX_TOKENS),
    # Size-matched frontier model: the honest comparison for a 9B adapter.
    Baseline("claude-haiku-4-5", "api", "claude-haiku-4-5", DEFAULT_FEW_SHOT, 0.0, API_MAX_TOKENS),
)

BASELINES: dict[str, Baseline] = {baseline.name: baseline for baseline in _BASELINES}


def get_baseline(name: str) -> Baseline:
    """Look up a baseline by the name the command line takes.

    Params:
        name: A key of BASELINES.

    Returns:
        The baseline.

    Raises:
        KeyError: If no baseline goes by that name.
    """
    if name not in BASELINES:
        raise KeyError(f"Unknown baseline '{name}'. Known: {', '.join(sorted(BASELINES))}")
    return BASELINES[name]
