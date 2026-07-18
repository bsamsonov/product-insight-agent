---
name: golden-set-builder
description: >
  Build or extend a RAG golden-set eval file (poc_main/packages/evals/data/golden_set_*.jsonl) for this
  POC, grounded in the real corpus (poc_main/data/raw/reviews.jsonl). Use whenever the user asks to
  create, write, regenerate, extend, or fix a golden set / eval set / eval questions, to add
  expected_chunk_ids, or to make eval cases match a new dataset. Produces JSONL cases whose
  expected_answer_substrings provably occur in the corpus and whose expected_chunk_ids are real
  indexed chunk ids — never invented from memory. Always run the validator before declaring done.
---

# Golden-set builder

A golden set is the ground truth for `poc_main/scripts/run_eval.py`. Its quality is decided by one
thing: **is every expected value actually grounded in the corpus?** Two metrics depend on it:

- `expected_answer_substrings` → substring match + `context_recall`. The substring is checked
  against the **LLM answer text**. An answer can only contain a word if that word is common in
  the retrieved reviews. So substrings must be frequent, generic, lowercase content words from
  the relevant reviews — not clever paraphrases.
- `expected_chunk_ids` → `citation_precision`. These must be the **real chunk ids** the indexer
  produced. A chunk id is `f"{review_id}__c{position}"`, e.g. `B0BYFLBC89__7__c0`. Inventing
  them gives a silently broken metric (precision 0), which is exactly the bug in the current
  sets — their `expected_chunk_ids` are all empty.

The hard part (grounding) is **mechanical**, not creative. Do not reason about the corpus from
memory — query it. `poc_main/scripts/golden_set_tools.py` turns every grounding step into a command.

## Fast path (cheapest — start here)

When budget matters, do **not** hand-write 35 cases with an LLM. Generate a grounded draft for
zero LLM cost, then spend a small model only on naturalizing phrasing:

```bash
uv run python poc_main/scripts/golden_set_tools.py draft          # → poc_main/packages/evals/data/golden_set_v2.jsonl
uv run python poc_main/scripts/golden_set_tools.py validate golden_set_v2
```
`draft` emits one templated question per product with grounded substrings and real chunk ids, so
the set is valid immediately. The only weakness is phrasing: when a product's top word is an
adjective the template reads awkwardly (e.g. "the small of the …", "quality and leggings of …").
Fix that with a single cheap pass — a Haiku/small-model run that **rewrites only the `question`
field** to read naturally, leaving `expected_answer_substrings`, `expected_chunk_ids`, ids and
metadata untouched (they are already grounded; changing them reintroduces risk). Re-run
`validate` afterwards. For a quick POC the raw draft is often good enough as-is.

Use the full manual workflow below only when you need hand-crafted, high-quality questions.

## Workflow

Create one todo per step.

### 1. Confirm the corpus is current
`poc_main/data/raw/reviews.jsonl` is the source of truth. Check it matches the intended domain:
```bash
uv run python poc_main/scripts/golden_set_tools.py products --top 30 --min-reviews 20
```
This lists candidate products (most reviews first) with category, review count and the most
frequent content words. If the products look like the wrong domain, stop — the corpus must be
rebuilt (`poc_main/scripts/download_dataset.py`) and reindexed first.

### 2. Pick anchor products
Build most questions **anchored to a specific product** (1 product, sometimes 2–3 for a
comparison/theme). Anchoring is what makes `expected_chunk_ids` well-defined: the gold chunk set
is simply *all chunks of the anchor product(s)*. Prefer products with ≥40 reviews so retrieval
has signal and the frequent-words list is stable.

Aim for ~30–40 cases spread across distinct products and categories (apparel, fishing, camping,
fitness, accessories…), so the set exercises the whole corpus, not one niche.

### 3. For each product, pull grounded material
```bash
uv run python poc_main/scripts/golden_set_tools.py product B0BYFLBC89
```
Gives the title, review count, the **frequent content words** (your substring menu — `df` is how
many reviews use the word) and the **full JSON array of that product's chunk ids** (your
`expected_chunk_ids`). For a cross-product theme question, get shared vocabulary with:
```bash
uv run python poc_main/scripts/golden_set_tools.py terms B0BYFLBC89 B09YSNSLNQ
```

### 4. Write the cases
One JSON object per line in `poc_main/packages/evals/data/golden_set_<name>.jsonl`. Schema:
```json
{"id":"q001","question":"...","expected_answer_substrings":["...","..."],"expected_chunk_ids":["..."],"filters":{},"metadata":{"category":"...","product_id":"..."}}
```
Rules that keep cases grounded:
- **question** — natural, answerable purely from reviews of the anchor product. Good shapes:
  "What do reviewers say about <aspect> of <product>?", "Are there complaints about <aspect>?",
  "How is the <fit/durability/value> of <product>?". Avoid questions needing outside knowledge,
  exact numbers, or a single reviewer's anecdote.
- **expected_answer_substrings** — 2–4 items, lowercase, taken from the product's frequent-words
  list (high `df`), and genuinely on-topic for the question. They match as plain substrings, so
  prefer stems that survive morphology: `"cushion"` matches *cushioned/cushioning*, `"fit"`
  matches *fits/fitting*. Pick words a good summary of those reviews would naturally contain.
- **expected_chunk_ids** — paste the array from step 3 (all chunks of the anchor product). For a
  multi-product question, concatenate their arrays.
- **id** — unique, zero-padded (`q001`…). **metadata.product_id** — the anchor, for traceability.

See `references/examples.md` for worked cases.

### 5. Validate — required gate
```bash
uv run python poc_main/scripts/golden_set_tools.py validate poc_main/packages/evals/data/golden_set_<name>.jsonl
```
It checks every case against the live corpus: substrings occur (with counts), chunk ids exist,
ids are unique, schema intact. **It exits non-zero on any error — fix and re-run until it passes
with 0 errors.** Treat `WARN substring rare (Nx)` as a nudge to pick a more common word.

### 6. Wire it in (only if asked to make it the default)
Point `_DEFAULT_EVAL_SET` in `poc_main/scripts/run_eval.py` at the new file, then do a real run after the
corpus is indexed into Qdrant (`poc_main/scripts/index.py --recreate`):
```bash
uv run python poc_main/scripts/run_eval.py --golden-set golden_set_<name> \
  --metrics citation_precision,groundedness_heuristic
```
A near-zero `substring_match_rate` or `citation_precision` after this means the cases are not
really grounded — go back to step 3. The validator proves the values exist in the corpus; the
eval run proves they survive retrieval + generation.

## Anti-patterns (why sets break)
- Substrings invented as paraphrases ("the customer experience was positive") — won't appear in a
  short factual answer. Use single frequent words.
- Empty `expected_chunk_ids` — makes `citation_precision` meaningless. Anchor to a product and
  fill them.
- Questions about products with few reviews — retrieval is noisy, metrics swing.
- Skipping the validator — the one step that guarantees grounding.
