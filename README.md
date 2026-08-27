# Liberty Mutual LLM Project

Instruction following is a key capability for LLMs to reason over complex tasks. This project contains a synthetic data generation pipeline for training a model for multi-hop instruction following. The generated data is then used to fine-tune an open source model with [Tinker](https://tinker-docs.thinkingmachines.ai), first with supervised fine-tuning and then with reinforcement learning.

## Quick start

Set `OPENAI_API_KEY` (data generation) and `TINKER_API_KEY` (training) in `.env`.

```bash
cp .env.example .env
```

Setup the virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

Log into HuggingFace and follow steps.

```
hf auth login
```

Generate data, then train.

```bash
python3 -m scripts.generate        # generate rows into artifacts/runs/
python3 -m scripts.prepare_data    # convert a run to conversations + token stats
python3 -m scripts.train_sft
python3 -m scripts.train_rl
```

## Layout

```
src/
  paths.py        every derived location, computed once
  config/         models and prices, generation knobs, training hyperparameters
  schema/         pydantic shapes shared by every stage
  llm/            thread-local OpenAI clients and token accounting
  generation/     synthetic data generation and its rubric gate
  training/       answer format, reward metrics, SFT, RL, export
  evaluation/     sampling, LLM judge, and the scoring run
scripts/          command line entry points, one per stage
docs/             pipeline flow and the generation rubric
artifacts/        all derived output: runs/, checkpoints/, results/
```

Imports flow one way: `schema` and `config` are read by everything, `generation` and `training` sit on top of them, and `evaluation` sits on top of `training`. Nothing under `src/` parses arguments or prints a report, so each stage can be driven from a script or imported directly.

Everything under `artifacts/` is regenerable and gitignored, apart from `artifacts/runs/sample_run.json`, a small committed run kept as a sample of the generated format. The full datasets live on the Hub instead of in git.

## Generation

`python3 -m scripts.generate` builds `NUM_ROWS` rows and writes them, with the settings and token cost that produced them, to `artifacts/runs/run_<timestamp>.json`. Each row is seeded from an existing multi-hop dataset and passed through a rubric gate before it is kept. See [docs/pipeline.md](docs/pipeline.md) for the stage by stage flow and the rubric itself.

`UNANSWERABLE_FRACTION` of each run is built with no informative chunk at all, so the model also learns to cite nothing and decline when the pool cannot support the instruction. Both knobs can be set per run:

```bash
python3 -m scripts.generate --num-rows 750 --unanswerable-fraction 1.0 --out artifacts/runs/unanswerable_750.json
```

Push a finished run to the Hub as a dataset:

```bash
python3 -m scripts.upload_to_hf --file artifacts/runs/combined_3000.json
```

## Training

Both trainers run locally and send the compute to Tinker. Everything they read lives in `src/config/training.py`, and any field can be overridden per run:

```bash
python3 -m scripts.train_sft learning_rate=2e-4 num_epochs=1
python3 -m scripts.train_rl group_size=16 max_tokens=3072
```

**Answer format.** `src/training/format.py` defines the one format both stages use: an `Informative IDs` line naming the gold search results, then a rationale, then a response grounded in those results. SFT teaches it from the teacher rationales; RL grades against it.

**SFT** (`src/training/sft.py`) trains a LoRA adapter on the generated conversations, applying loss to assistant tokens only. It builds `artifacts/runs/sft/conversations.jsonl` on demand and holds out `TEST_SIZE` rows for held out NLL.

**RL** (`src/training/rl.py`) runs GRPO from the SFT checkpoint. Each environment shows one instruction and its noisy search pool, and `src/training/metrics.py` scores the sampled answer on citation F1 against the gold chunk IDs, coverage of the per-hop answers, and a penalty for parroting distractor-only vocabulary. Reward components are logged per step so they can be monitored individually.

**Metrics.** Both trainers log to Weights & Biases when `WANDB_API_KEY` is set, under the project named by `WANDB_PROJECT` in `src/config/training.py`. The full resolved config is uploaded with the run, so command line overrides show up in the dashboard without being named in the run title. Set `WANDB_PROJECT` to `None` to turn the upload off.

SFT reports `train_nll`, `train_mean_bpb`, `learning_rate`, and held out `test/nll` and `test/bpb` on the validation rows. RL reports the reward components from `src/training/metrics.py` split by row type, so `env/answerable/` and `env/unanswerable/` are separate series: an unanswerable group whose samples all score alike contributes no advantage, and that shows up as a flat `env/unanswerable/` curve rather than hiding inside a blended average.

Without a key the runs are unaffected and every metric still lands in `metrics.jsonl` under the run's log path.

**Export.**

```bash
python3 -m scripts.export --run rl --push
```

Metrics, rollout logs, and checkpoints are written to `artifacts/checkpoints/sft` and `artifacts/checkpoints/rl`; `metrics.jsonl` in either directory is the file to watch during a run.

## Evaluation

`src/evaluation/` scores a model on the 100 rows held out of every training split, so the base model, the SFT checkpoint, and the RL checkpoint are directly comparable.

```bash
python3 -m scripts.evaluate --run base            # baseline before training
python3 -m scripts.evaluate --run sft
python3 -m scripts.evaluate --run rl
python3 -m scripts.evaluate --run rl --no-judge   # programmatic metrics only, no API cost
```

Two graders run over the same responses:

- **Programmatic** (`src/training/metrics.py`): citation F1 against the gold chunk IDs, per-hop answer coverage, and distractor leakage. This is the same function RL optimizes, which is why it lives alongside the trainers rather than in the evaluation package.
- **LLM judge** (`src/evaluation/judge.py`, prompt in `src/evaluation/prompts.py`): a strong model scores chunk selection, rationale quality, hop completeness, answer grounding, and distractor resistance as 0 or 1 with a justification. Set the model with `JUDGE_MODEL` in `src/config/models.py`; any model used there needs an entry in `TOKEN_PRICE`.

Aggregates print to the terminal and every response, score, and justification lands in `artifacts/results/<run>.json`.
