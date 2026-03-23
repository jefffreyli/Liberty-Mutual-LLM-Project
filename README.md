# Liberty Mutual LLM Project

Instruction following is a key capability for LLMs to reason over complex tasks. This project contains a synthetic data generation pipeline for training a model for multi-hop instruction following. The generated data is used to train an open source model for multi-hop instruction following.

## Quick start

Set `OPENAI_API_KEY` in `.env` for LLM access.

```bash
cp .env.example .env
```

Setup the virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

Run the main script.

```bash
python src/main.py
```

## Project structure

```
src/
├── data/
│   ├── generator.py
│   ├── evaluator.py # rubric-based quality gate
│   ├── noise.py
│   └── prompts.py
├── schema/
│   ├── core.py
│   ├── evaluation.py
│   ├── generation.py
│   └── schemas.py
├── utils/
│   ├── llm_client.py
├── config.py
├── main.py # CLI entry point
└── requirements.txt
```