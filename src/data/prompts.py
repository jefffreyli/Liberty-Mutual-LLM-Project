# Prompt for generating multi-hop instructions.

GENERAL_INSTRUCTION_PROMPT = """You are an expert dataset author. Generate a clear multi-step instruction about common, general-knowledge topics that a non-specialist can manually verify.

Use this seed example as inspiration for style and structure, but do NOT copy it:
- Seed instruction: {seed_instruction}
- Seed input: {seed_input}
- Seed output: {seed_output}

The instruction must:
- Require at least 2 reasoning steps (hops) to complete correctly
- Use everyday domains (e.g. travel planning, school tasks, budgeting, scheduling, cooking, health habits, consumer decisions, workplace logistics)
- Have a clear, factual target output
- Avoid niche technical jargon and domain-specific legal/regulatory detail
- Be meaningfully different from the seed example in topic and wording

Provide:
1. The multi-hop instruction
2. The decomposition into single-hop sub-instructions (each with its answer)
3. Informative paragraphs (2-4) that contain the information needed to complete each sub-instruction
4. The final answer"""

INSURANCE_INSTRUCTION_PROMPT = """You are an expert insurance knowledge engineer. Generate a challenging multi-hop instruction that requires reasoning across multiple insurance concepts.

The instruction must:
- Require at least 2 reasoning steps (hops) to complete correctly
- Cover real insurance topics (e.g. underwriting, claims, policy terms, regulations, actuarial concepts)
- Have a clear, factual target output

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

Instruction: {instruction}

Here are the informative paragraphs the instruction is based on:
{informative_text}

Generate exactly {n} distractor paragraphs. Each distractor must be:
- A "neighboring concept": topically adjacent to the informative paragraphs but covering a DIFFERENT aspect, category, jurisdiction, policy type, or time period
  Examples: if the informative text discusses commercial general liability, a distractor might cover professional liability or product liability; if it discusses California regulations, a distractor might cover New York regulations
- Plausible as a real search result that a retrieval system might return for the same query
- NOT a factual contradiction or false rewrite of anything stated in the informative paragraphs
- NOT designed to trick the model with wrong versions of the truth

The goal is to test the model's ability to identify which retrieved passages are actually relevant to completing the instruction, NOT to test whether the model can detect falsehoods.

Each paragraph needs a realistic title and body text."""

# Prompt for evaluating the training example using the rubric.
RUBRIC_PROMPT = """You are evaluating a synthetic training example for a search-augmented multi-hop QA system. Score each metric as 0 (fail) or 1 (pass) with a brief justification.

## Training Example

**Instruction:** {instruction}

**Decomposition:**
{decomposition_text}

**Search Pool:**
{search_pool_text}

**Informative Paragraph IDs:** {informative_ids}

**Response:** {response}

## Evaluation Rubric

1. **logical_necessity**: Every informative paragraph is required to complete the instruction. Removing any single informative paragraph would make the instruction unsatisfiable. Score 0 if any informative paragraph is redundant or unnecessary.

2. **distractor_plausibility**: Distractor paragraphs are topically related to the instruction and share high semantic similarity with the task domain. They should be genuinely plausible retrieval results a search engine might return. Score 0 if any distractor is obviously irrelevant or off-topic.

3. **non_contradiction**: No distractor paragraph contains false versions of facts stated in the informative paragraphs. Distractors should cover neighboring concepts (different jurisdictions, policy types, time periods) rather than contradicting the truth. Score 0 if any distractor directly contradicts or falsely rewrites informative content.

4. **answer_grounding**: The response is fully supported by and grounded in ONLY the informative paragraphs. It contains no hallucinated facts or outside knowledge. Score 0 if the response contains claims not traceable to the informative paragraphs."""

__all__ = [
    "GENERAL_INSTRUCTION_PROMPT",
    "INSURANCE_INSTRUCTION_PROMPT",
    "RATIONALE_PROMPT",
    "RESPONSE_PROMPT",
    "DISTRACTOR_PROMPT",
    "RUBRIC_PROMPT",
]
