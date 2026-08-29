"""Judge prompt used to grade a model response against the row it was generated from, scoring each
rubric metric as 0 or 1 with a justification. The decomposition section and the hop completeness
metric are filled in only for datasets that ship per hop ground truth; everything else grades the
four metrics that need nothing but the search pool and the gold IDs.
"""

RESPONSE_JUDGE_PROMPT = """You are grading a model's answer to an instruction that had to be completed using a noisy search pool. Score each metric as 0 (fail) or 1 (pass) with a brief justification.

## The Task Given to the Model

**Instruction:** {instruction}

**Search Pool:**
{search_pool_text}

**Informative Paragraph IDs (ground truth, hidden from the model):** {informative_ids}
{decomposition_section}
**Reference Answer ({reference_label}):** {reference_response}

## The Model's Answer

**Cited Informative IDs:** {cited_ids}

**Rationale:** {rationale}

**Response:** {response}

## Evaluation Rubric

- **chunk_selection**: The cited IDs match the ground truth informative IDs. Score 0 if the model missed an informative paragraph or cited a distracting one.

- **rationale_quality**: The rationale gives a correct reason why each cited paragraph is informative and why the remaining ones are distracting (neighboring entity, different time period, different jurisdiction, different policy type). Score 0 if the reasoning is generic, absent, or wrong about a paragraph.
{hop_rubric}
- **answer_grounding**: Every claim in the response is traceable to the informative paragraphs. Score 0 if the response invents facts or relies on outside knowledge, even if the claim happens to be true.

- **distractor_resistance**: The response does not treat content from distracting paragraphs as fact. Score 0 if it repeats or reasons from a distractor.

Judge only what the model actually wrote. A response that reaches the right conclusion by the wrong route fails the metric that covers that route."""

DECOMPOSITION_SECTION = """
**Required Reasoning Steps (ground truth, hidden from the model):**
{decomposition_text}
"""

HOP_COMPLETENESS_RUBRIC = """
- **hop_completeness**: The response resolves every required reasoning step, not just the easiest one. Score 0 if any step is skipped, left implicit, or answered incorrectly.
"""
