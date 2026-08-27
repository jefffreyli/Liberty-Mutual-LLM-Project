# Generation pipeline

How one training row is built. Every arrow marked `-> LLM ->` is a call to the teacher model
configured as `DEFAULT_MODEL` in `src/config/models.py`.

```
GENERAL_INSTRUCTION_PROMPT -> LLM -> instruction bundle
-> build informative chunks
-> DISTRACTOR_PROMPT -> LLM -> distractor chunks
-> merge + shuffle into search pool (assign IDs)
-> map decomposition steps to support paragraph IDs
-> RATIONALE_PROMPT -> LLM -> rationale
-> RESPONSE_PROMPT -> LLM -> grounded response
-> assemble TrainingRow
-> RUBRIC_PROMPT -> LLM -> 4 metric scores
-> code gate (hard-pass metrics + threshold)
-> fail: regenerate (up to MAX_RETRIES_PER_ROW) / pass: keep row
-> repeat for NUM_ROWS rows
-> write artifacts/runs/run_<timestamp>.json
```

The prompts live in `src/generation/prompts.py`, the per stage calls in
`src/generation/instruction.py` and `src/generation/noise.py`, and the loop that drives them in
`src/generation/generator.py`.

## Unanswerable rows

`UNANSWERABLE_FRACTION` of every run is built with no informative chunk at all, so the model sees
pools where citing nothing is the correct answer. Without them a discriminator trained only on
answerable rows learns to always cite a few chunks, and the empty gold branch in
`citation_scores` never fires during training. The approach follows the grounding sampling in
[SAIL: Search-Augmented Instruction Learning](https://arxiv.org/abs/2305.15225), which draws
zero to three search results per training case so the model handles degenerate grounding.

These rows are built by `src/generation/unanswerable.py`, which runs the normal instruction
stage and then throws the informative paragraphs away:

```
GENERAL_INSTRUCTION_PROMPT -> LLM -> instruction bundle
-> DISTRACTOR_PROMPT -> LLM -> NUM_UNANSWERABLE_DISTRACTORS distractor chunks
-> DISCARD the informative paragraphs
-> shuffle distractors into search pool (assign IDs)
-> UNANSWERABLE_RATIONALE_PROMPT -> LLM -> rationale rejecting every ID
-> UNANSWERABLE_RESPONSE_PROMPT -> LLM -> response that declines to answer
-> assemble TrainingRow with an empty decomposition
-> UNANSWERABLE_RUBRIC_PROMPT -> LLM -> 3 metric scores
-> code gate (hard-pass metrics + threshold)
```

The distractors are generated against the informative paragraphs before those paragraphs are
dropped, so the pool sits next to a real answer without containing it. That is the shape of a
realistic retrieval failure, where the retriever returned topically adjacent documents and none
of them are enough.

The decomposition is left empty because its per hop answers are not recoverable from the pool.
`answer_coverage` returns 1.0 for a row with no hops, matching how `citation_scores` already
treats an empty gold set, so an abstention that cites nothing scores near 1.0 while an answer
that cites chunks anyway scores near 0.5.

**No support.** No search result, alone or combined with the others, provides the information
needed to complete the instruction.

**Distractor plausibility.** The results are topically related and plausible retrieval results,
so the pool is not trivially rejectable.

**Abstention correctness.** The response declines, names what is missing, and does not fabricate
an answer or import outside knowledge.

## Quality rubric

The gate in `src/generation/rubric.py` scores every candidate row on four metrics, each 0 or 1
with a justification. Weights, the pass threshold, and which metrics are hard constraints are set
in `src/config/generation.py`.

**Logical necessity.** Every informative paragraph is required to complete the instruction.
Removing any single informative paragraph would make the instruction unsatisfiable.

**Distractor plausibility.** Distractor paragraphs are topically related to the instruction and
share high semantic similarity with the task domain. They should be genuinely plausible retrieval
results a search engine might return.

**Non-contradiction.** No distractor paragraph contains false versions of facts stated in the
informative paragraphs. Distractors should cover neighboring concepts (different jurisdictions,
policy types, time periods) rather than contradicting the truth.

**Answer grounding.** The response is fully supported by and grounded in ONLY the informative
paragraphs. It contains no hallucinated facts or outside knowledge.

## Seeding

Generating from scratch collapses toward a narrow set of topics, so each row is seeded from an
example in an existing multi-hop dataset, set by `SEED_DATASET_NAME` in
`src/config/generation.py`. The approach follows [SELF-INSTRUCT: Aligning Language Models with
Self-Generated Instructions](https://www.alphaxiv.org/abs/2212.10560). `src/generation/seed_loader.py`
loads the seed set and hands one example per row to the generator.

Rows held out for evaluation come from the same generated pool rather than from the seed
datasets, so `src/evaluation` scores the base model, the SFT checkpoint, and the RL checkpoint on
identical rows.
