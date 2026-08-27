"""Every model the pipeline calls through the OpenAI API, and what those calls cost. The teacher
that generates and grades data, the judge that scores evaluation responses, and the sampling
settings for the model under evaluation all live here so pricing stays in one table.
"""

# Teacher. Writes the instructions, paragraphs, rationales, and responses, and
# scores candidate rows against the generation rubric.
DEFAULT_MODEL = "gpt-4o"

# Judge. A strong model is worth the cost here because it is the only grader that
# reads an evaluation response as a whole.
JUDGE_MODEL = "gpt-5.5"
JUDGE_WORKERS = 8  # parallel judge calls, each on its own thread-local client

# Sampling from the model under evaluation.
SAMPLE_TEMPERATURE = 0.0  # greedy, so repeated runs are comparable
SAMPLE_WORKERS = 16  # concurrent rollouts

# Dollars per million tokens. Every model named above needs an entry, because the
# token tracker prices each call as it is made.
TOKEN_PRICE = {
    "gpt-4o": {
        "input": 2.5,
        "output": 10.0,
    },
    "gpt-5.5": {
        "input": 5.0,
        "output": 30.0,
    },
}
