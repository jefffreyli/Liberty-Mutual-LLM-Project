"""Teacher model prompts for every generation stage: the multi-hop instruction and its
informative paragraphs, the distractors, the rationale, the grounded response, and the rubric
used to accept or reject a finished row.
"""

# Prompt for generating multi-hop instructions.
SEED_EXAMPLE_BLOCK = """Use this seed example as inspiration for style and structure, but do NOT copy it:
- Seed instruction: {seed_instruction}
- Seed paragraphs: {seed_paragraphs}
- Seed decomposition: {seed_decomposition}
- Seed output: {seed_output}
The generated instruction must be meaningfully different from the seed example in topic and wording.
Note: the seed may be phrased as a question — your output must still be an imperative instruction, not a question."""

NO_SEED_INSTRUCTION_BLOCK = """No seed example is provided. Generate a multi-hop instruction from scratch without copying any prior example."""

GENERAL_INSTRUCTION_PROMPT = """You are an expert dataset author. Generate a clear multi-step instruction about common, general-knowledge topics that a non-specialist can manually verify.

{seed_block}

The instruction must:
- Be phrased as an imperative command, NOT a question. Use action verbs such as "Determine", "Find", "Calculate", "Identify", "Explain", "Compare", "List", "Summarize", "Plan", or "Recommend". Never start with "What", "Who", "Which", "When", "Where", or "How".
- Require at least 2 reasoning steps (hops) to complete correctly
- Use everyday domains (e.g. travel planning, school tasks, budgeting, scheduling, cooking, health habits, consumer decisions, workplace logistics)
- Have a clear, factual target output
- Avoid niche technical jargon and domain-specific legal/regulatory detail

Each sub-instruction in the decomposition must also be phrased as an imperative command, not a question.

Provide:
1. The multi-hop instruction
2. The decomposition into single-hop sub-instructions (each with its answer)
3. Informative paragraphs (2-4) that contain the information needed to complete each sub-instruction
4. The final answer"""

# Prompt for generating the rationale followed by the LLM from the search pool of informative and distracting chunks.
RATIONALE_PROMPT = """You are analyzing a search pool for a multi-hop QA task.

Instruction: {instruction}

Search pool:
{search_pool_text}

Informative paragraph IDs: {informative_ids}

Write a rationale that:
1. Identifies which search result IDs are informative and which are distracting
2. Explains WHY each distracting chunk is not useful for completing the instruction (e.g. covers a neighboring concept, different jurisdiction, different policy type, different time period)
3. Traces the reasoning chain through the informative paragraphs to reach the answer"""

# Prompt for generating the response grounded only in the informative paragraphs.
RESPONSE_PROMPT = """You are completing an instruction using ONLY the provided informative paragraphs. Do not use any outside knowledge.

Instruction: {instruction}

Informative paragraphs:
{informative_text}

Write a comprehensive response grounded entirely in the informative paragraphs above. Cite specific details from the paragraphs."""

# Prompt for generating the distractor paragraphs.
DISTRACTOR_PROMPT = """You are generating distractor paragraphs for a search-augmented QA training dataset.

Instruction the user is trying to complete: {instruction}

Here are the informative paragraphs that actually answer it:
{informative_text}

Generate exactly {n} distractor paragraphs. Each distractor must be:
- About the SAME aspect or subtopic as one of the informative paragraphs, not a different one. Take what an informative paragraph covers and shift exactly one parameter: the jurisdiction, the year, the policy tier, the population, the product variant, the unit of measurement
  Example: if an informative paragraph gives the 2024 federal rate, a distractor gives the 2019 rate, or the state rate, or the rate for a different filing category
- Written using the SAME vocabulary and phrasing as the instruction above, so that counting shared words with the instruction cannot separate it from the informative paragraphs. Reuse the instruction's own terms freely
- Genuinely unable to complete the instruction, because the one shifted parameter makes it inapplicable
- NOT a factual contradiction or false rewrite of anything stated in the informative paragraphs. The facts must be true of the case they describe, just the wrong case

A reader must have to notice the shifted parameter to reject the paragraph. Surface similarity to the instruction is the goal, not a signal of relevance.

Each paragraph needs a realistic title and body text."""

# Prompt for generating chunks that state a false version of a gold fact, so the
# model has to prefer the mutually consistent informative cluster.
CONTRADICTORY_PROMPT = """You are generating contradictory paragraphs for a search-augmented QA training dataset.

Instruction: {instruction}

Here are the informative paragraphs, which are the ground truth and are mutually consistent:
{informative_text}

Generate exactly {n} contradictory paragraphs. Each one must:
- Assert a FALSE version of one specific fact stated in the informative paragraphs: a different number, date, name, threshold, or outcome for the SAME case
- Describe the same jurisdiction, period and category as the fact it contradicts, so it cannot be dismissed as merely covering a neighboring case. It has to be wrong, not off topic
- Read as an ordinary, confident retrieval result. No hedging and no signal that it is unreliable
- Contradict the informative paragraphs only, never another contradictory paragraph

For each one, report `contradicted_fact`: the true statement from the informative paragraphs that it contradicts, quoted or closely paraphrased.

The informative paragraphs agree with each other and each contradictory paragraph is an outlier against them. That consistency is the only cue distinguishing them, which is exactly the signal the model must learn to use."""

# Prompt for evaluating the training example using the rubric.
RUBRIC_PROMPT = """You are evaluating a synthetic training example for a search-augmented multi-hop QA system. Score each metric as 0 (fail) or 1 (pass) with a brief justification.

## Training Example

**Instruction:** {instruction}

**Decomposition:**
{decomposition_text}

**Search Pool:**
{search_pool_text}

**Informative Paragraph IDs:** {informative_ids}
**Contradictory Paragraph IDs:** {contradictory_ids}

**Response:** {response}

## Evaluation Rubric

1. **logical_necessity**: Every informative paragraph is required to complete the instruction. Removing any single informative paragraph would make the instruction unsatisfiable. Score 0 if any informative paragraph is redundant or unnecessary.

2. **distractor_plausibility**: Distractor paragraphs are topically related to the instruction and share high semantic similarity with the task domain. They should be genuinely plausible retrieval results a search engine might return. Score 0 if any distractor is obviously irrelevant or off-topic.

3. **non_contradiction**: No paragraph listed as a plain distractor contains a false version of a fact stated in the informative paragraphs. Plain distractors shift one parameter (jurisdiction, period, category) and stay true of the case they describe. Paragraphs listed as contradictory are exempt from this metric and are scored by contradiction_validity instead. Score 0 if a plain distractor falsely rewrites informative content.

5. **contradiction_validity**: Every paragraph listed as contradictory asserts a false version of a fact the informative paragraphs state, for the same case rather than a neighboring one, and the informative paragraphs remain mutually consistent with each other. Score 1 when the row lists no contradictory paragraphs. Score 0 if a contradictory paragraph is merely off topic, contradicts another contradictory paragraph, or if the informative paragraphs disagree among themselves.

4. **answer_grounding**: The response is fully supported by and grounded in ONLY the informative paragraphs. It contains no hallucinated facts or outside knowledge. Score 0 if the response contains claims not traceable to the informative paragraphs."""

# Prompts for unanswerable rows, whose search pool holds no informative chunk.

UNANSWERABLE_RATIONALE_PROMPT = """You are analyzing a search pool for a multi-hop instruction-following task.

Instruction: {instruction}

Search pool:
{search_pool_text}

NONE of these search results are informative. Every one of them is a distracting retrieval result: topically adjacent to the instruction but missing the information needed to complete it.

Write a rationale that:
1. States that no search result in the pool is informative
2. Explains for EACH search result ID why it is distracting (which neighboring concept it covers instead, and what it fails to provide)
3. Names the specific information that would be required to complete the instruction and is absent from the pool

Do not speculate about what the answer might be. Do not use outside knowledge."""

UNANSWERABLE_RESPONSE_PROMPT = """You are completing an instruction using ONLY the provided search results. None of them contain the information needed.

Instruction: {instruction}

Search results:
{search_pool_text}

Write a short response that:
- States plainly that the search results do not contain the information needed to complete the instruction
- Names the specific information that is missing, phrased in terms of the instruction itself
- Does NOT answer the instruction from outside knowledge, and does NOT guess

Keep it to a few sentences. Describe the gap using the language of the instruction rather than restating the content of the search results."""

UNANSWERABLE_RUBRIC_PROMPT = """You are evaluating a synthetic training example for a search-augmented multi-hop QA system. This example is deliberately UNANSWERABLE: the search pool should contain no result that helps complete the instruction, and the response should decline to answer. Score each metric as 0 (fail) or 1 (pass) with a brief justification.

## Training Example

**Instruction:** {instruction}

**Search Pool:**
{search_pool_text}

**Response:** {response}

## Evaluation Rubric

1. **no_support**: No search result, alone or in combination with the others, provides the information needed to complete the instruction. Score 0 if any single result or any combination of results would let a careful reader complete the instruction.

2. **distractor_plausibility**: The search results are topically related to the instruction and are genuinely plausible retrieval results a search engine might return for it. Score 0 if any result is obviously irrelevant or off topic, which would make the pool trivially easy to reject.

3. **abstention_correctness**: The response states that the search results do not support completing the instruction, identifies what information is missing, and does not fabricate an answer or import outside knowledge. Score 0 if the response attempts to answer the instruction, guesses, or hedges into a partial answer."""
