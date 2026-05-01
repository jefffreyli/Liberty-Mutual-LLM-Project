# Liberty Mutual LLM Project

Instruction following is a key capability for LLMs to reason over complex tasks. This project contains a synthetic data generation pipeline for training a model for multi-hop instruction following. The generated data is used to train an open source model for multi-hop instruction following.

## Quick start

Set `OPENAI_API_KEY` in `.env` for LLM access.

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

Run the main script.

```bash
python3 -m src.main
python3 -m src.training.train
```

