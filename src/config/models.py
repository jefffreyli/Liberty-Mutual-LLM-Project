"""Every model the pipeline calls, which provider serves it, and what its calls cost. The teacher
that generates and grades data, the judges that score evaluation responses, and the API baselines
the fine-tuned model is compared against all resolve through the one table here, so a model can
never be routed to a provider without a price or priced without a route.
"""

# Teacher. Writes the instructions, paragraphs, rationales, and responses, and
# scores candidate rows against the generation rubric.
DEFAULT_MODEL = "gpt-4o"

# Judges. A strong model is worth the cost here because it is the only grader
# that reads an evaluation response as a whole. Listing more than one grades
# every response with each of them: the teacher, the default judge, and the
# strongest API baselines are all OpenAI models, so a second judge from another
# family is what keeps a cross-model comparison from being graded by a relative.
JUDGE_MODELS = ("gpt-5.5",)
JUDGE_WORKERS = 8  # parallel judge calls, each on its own thread-local client
# A judge call that fails returns no verdict and the row is dropped from the
# aggregate, so a judge failing often would report a mean over whichever rows
# happened to succeed. Below this share of rows graded, the run fails instead.
MIN_JUDGE_COVERAGE = 0.9

# Output cap for structured generation. Only the Anthropic API requires one, but
# a verdict with five justifications needs room, and on a thinking model the
# reasoning is billed against the same budget.
STRUCTURED_MAX_TOKENS = 16000

# Sampling from the model under evaluation. Each baseline in
# src/evaluation/baselines.py sets its own temperature, because the reasoning
# models reject the parameter outright rather than ignoring it.
SAMPLE_WORKERS = 16  # concurrent rollouts

# Every model named above or in src/evaluation/baselines.py needs an entry: the
# client registry routes on `provider` and the token tracker prices each call as
# it is made, both keyed by the model id the caller asked for. Prices are dollars
# per million tokens.
MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "input": 2.5,
        "output": 10.0,
    },
    "gpt-5.5": {
        "provider": "openai",
        "input": 5.0,
        "output": 30.0,
    },
    "claude-opus-5": {
        "provider": "anthropic",
        "input": 5.0,
        "output": 25.0,
    },
    "claude-haiku-4-5": {
        "provider": "anthropic",
        "input": 1.0,
        "output": 5.0,
    },
}
