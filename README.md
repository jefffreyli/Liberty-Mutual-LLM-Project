# Liberty Mutual LLM Project

Instruction following is a key capability for LLMs to reason over complex tasks. This project contains a synthetic data generation pipeline for training a model for multi-hop instruction following. The generated data is then used to fine-tune an open source model with [Tinker](https://tinker-docs.thinkingmachines.ai), first with supervised fine-tuning and then with reinforcement learning.

## Quick start

Set `OPENAI_API_KEY` (data generation) and `TINKER_API_KEY` (training) in `.env`. `ANTHROPIC_API_KEY` is only needed to score the Anthropic evaluation baselines or judge with them.

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
  llm/            thread-local provider clients, keyed by model, and token accounting
  generation/     synthetic data generation and its rubric gate
  training/       answer format, reward metrics, SFT, RL, export
  evaluation/     the baseline and benchmark registries, sampling, LLM judges, and the scoring run
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

`src/evaluation/` scores one model at a time on the rows held out of every training split, so every baseline answers the same instructions with the same prompt and the numbers are directly comparable.

```bash
python3 -m scripts.evaluate --run base            # untrained weights, zero shot
python3 -m scripts.evaluate --run base-fewshot    # untrained weights, shown the format
python3 -m scripts.evaluate --run sft
python3 -m scripts.evaluate --run rl
python3 -m scripts.evaluate --run qwen3.8-27b     # same family, three times the size
python3 -m scripts.evaluate --run gpt-4o          # the teacher that wrote the data
python3 -m scripts.evaluate --run claude-opus-5   # frontier ceiling
python3 -m scripts.evaluate --run rl --no-judge   # programmatic metrics only, no API cost
```

**Baselines.** `src/evaluation/baselines.py` is the registry of what `--run` accepts, and adding a comparison model means adding an entry there and, for a hosted model, its price in `src/config/models.py`. Each entry names either a set of Tinker weights or a hosted API model, along with how many exemplars to prompt with and what temperature to sample at; the reasoning models reject `temperature` outright, so it is per baseline rather than global. A Tinker baseline can also name its own `base_model`, which is how `qwen3.8-27b` samples a larger model of the same family without retraining anything.

`qwen3.8-27b` answers "did you just need a bigger model?", which no API baseline can: family, tokenizer, chat template, and serving stack all match the fine-tuned model, so size is close to the only thing that differs. Close, not equal, because Tinker serves no Qwen3.5 dense model above 9B, which makes this one three minor versions ahead as well as three times larger. That is a stronger opponent than a pure scale control rather than a weaker one, so beating it is a conservative result; losing to it cannot be attributed to size alone.

`base-fewshot` is the rung that keeps the ladder honest. The fine-tuned checkpoints were taught the answer format by training on it, so scoring them against a model that has never seen the format measures format compliance rather than capability. Every prompted baseline is shown the same exemplars, drawn by `src/evaluation/fewshot.py` from the *training* split so nothing leaks from the rows being scored, and always including one unanswerable example: a prompt whose every example cites chunks teaches the model to always cite something, and abstention is a fifth of the rows it is about to be graded on.

**Judges.** `JUDGE_MODELS` in `src/config/models.py` sets the default, and `--judge` overrides it and repeats:

```bash
python3 -m scripts.evaluate --run claude-opus-5 --judge gpt-5.5 --judge claude-opus-5
```

The teacher, the default judge, and the strongest API baselines are all OpenAI models, so a cross-model comparison graded only by `gpt-5.5` has the judge scoring its own family. Naming a second judge from another family grades every response twice and puts the disagreement in the results file. The programmatic metrics are family-neutral, which is why they lead the report.

Two graders run over the same responses:

- **Programmatic** (`src/training/metrics.py`): citation F1 against the gold chunk IDs, per-hop answer coverage, and distractor leakage. This is the same function RL optimizes, which is why it lives alongside the trainers rather than in the evaluation package.
- **LLM judge** (`src/evaluation/judge.py`, prompt in `src/evaluation/prompts.py`): a strong model scores chunk selection, rationale quality, hop completeness, answer grounding, and distractor resistance as 0 or 1 with a justification. Set the judges with `JUDGE_MODELS` in `src/config/models.py`; every model named anywhere in the pipeline needs an entry in `MODELS` there, which is what routes it to a provider and prices its calls.

Aggregates print to the terminal, overall and, on a benchmark that has both, split into answerable and unanswerable rows because the two are different tasks. Every response, score, and justification lands in `artifacts/results/<run>.json`.

**Benchmarks.** `--benchmark` picks the dataset instead of the model, so the same ladder can be scored on data the model was never trained on. `src/evaluation/benchmarks/` is the registry of what it accepts, and every entry adapts its dataset into the same `TrainingRow` shape, so one code path samples and grades all of them.

```bash
python3 -m scripts.evaluate --run rl --benchmark musique     # near transfer, and a seed source
python3 -m scripts.evaluate --run rl --benchmark hotpotqa    # retriever-drawn distractors
python3 -m scripts.evaluate --run rl --benchmark finqa       # financial filings, numeric answers
python3 -m scripts.evaluate --run rl --benchmark longbench   # the same task at 10x the context
python3 -m scripts.evaluate --run rl --benchmark browsecomp  # deep-research queries, mined hard negatives
```

The default, `synthetic`, is the generated test split and keeps writing to `artifacts/results/`; every other benchmark writes to `artifacts/results/<benchmark>/` and samples a seeded, stratified 300 rows. Rows are rejected before scoring if they would score well for the wrong reason: an empty decomposition makes `answer_coverage` return 1.0 for free, and a pool that is all gold or all distractor asks the model to decide nothing.

Two judge rubrics exist because `hop_completeness` needs real per hop ground truth. MuSiQue and the generated split ship it and are graded on all five metrics; the rest are graded on the four that need only the pool and the gold IDs. A judge that fails on more than a tenth of the rows fails the run rather than reporting a mean over whichever rows survived.

On `longbench` the gold answer is a paragraph id, so citation F1 is the number that matters and `answer_coverage` only checks whether the response names the paragraph it cited. Read `score` there as deflated by construction, not as a failure.

`browsecomp` is the one benchmark whose task is constructed rather than adapted. A native BrowseComp-Plus pool is about 87 documents averaging 40k characters, roughly 900k tokens, so `src/evaluation/benchmarks/browsecomp.py` builds a passage level pool instead: documents are split into 1200-character passages, a gold document's informative passage is the one carrying the verbatim answer, and the distractors are the hard-negative passages ranking highest on query overlap, which is what a retriever would have surfaced. Passages that carry the answer are never used as distractors, near duplicates are dropped, and the pool is shuffled before IDs are assigned so gold never sits at a fixed position. **Its scores are not comparable to the published BrowseComp-Plus leaderboard**, which measures a retriever and an agent over the full corpus.

Every field but `query_id` in that dataset is XOR obfuscated to keep it out of training crawls. It is de-obfuscated locally at load time and the plaintext only ever reaches `artifacts/`, which is gitignored. Do not publish it.
