# Worked golden-set cases

All values below come from `golden_set_tools.py` output, not from imagination. Each example
shows the command that produced the grounded material, then the resulting JSONL line.

## Example 1 — single-product, product feature

`uv run python poc_main/scripts/golden_set_tools.py product B0BYFLBC89` →
title "Reaction Tackle Braided Fishing Line", 60 reviews, frequent words include
`line, fishing, strong, braid, casting, knots, value`, and 60 chunk ids `B0BYFLBC89__0__c0 … __59__c0`.

```json
{"id":"q001","question":"What do reviewers say about the strength and casting of this braided fishing line?","expected_answer_substrings":["line","strong","casting"],"expected_chunk_ids":["B0BYFLBC89__0__c0","B0BYFLBC89__1__c0","...all 60..."],"filters":{},"metadata":{"category":"product_feature","product_id":"B0BYFLBC89"}}
```
Why it works: `strong`/`casting`/`line` are high-`df` words a summary will reuse; the gold chunks
are exactly this product's chunks, so any correct citation lands inside them.

## Example 2 — single-product, complaint / issue

`uv run python poc_main/scripts/golden_set_tools.py product B07D8CTXS7` → GSI camping percolator,
frequent words include `coffee, pot, percolator, lid, flimsy, plastic`.

```json
{"id":"q002","question":"Are there complaints about the build quality of this camping coffee percolator?","expected_answer_substrings":["plastic","flimsy","lid"],"expected_chunk_ids":["B07D8CTXS7__0__c0","...all chunks of B07D8CTXS7..."],"filters":{},"metadata":{"category":"product_issue","product_id":"B07D8CTXS7"}}
```
Note `flimsy` appeared in the frequent-words list — that is the signal a real complaint theme
exists in the corpus. Never assert a complaint the words don't support.

## Example 3 — single-product, fit/sizing (apparel)

`uv run python poc_main/scripts/golden_set_tools.py product B08R8PZX6D` → NFL jogger sweatpants,
frequent words `fit, size, pants, comfy, comfortable, small`.

```json
{"id":"q003","question":"How is the fit and comfort of these jogger sweatpants?","expected_answer_substrings":["fit","size","comfortable"],"expected_chunk_ids":["B08R8PZX6D__0__c0","...all chunks of B08R8PZX6D..."],"filters":{},"metadata":{"category":"product_feature","product_id":"B08R8PZX6D"}}
```

## Example 4 — cross-product theme

`uv run python poc_main/scripts/golden_set_tools.py terms B0BYFLBC89 B010LSTBWI` (two braided fishing
lines) → shared words `line, braid, strong, casting, knots`.

```json
{"id":"q010","question":"What qualities do customers value in braided fishing lines?","expected_answer_substrings":["line","braid","strong"],"expected_chunk_ids":["B0BYFLBC89__0__c0","...all chunks of B0BYFLBC89...","B010LSTBWI__0__c0","...all chunks of B010LSTBWI..."],"filters":{},"metadata":{"category":"theme","product_ids":["B0BYFLBC89","B010LSTBWI"]}}
```
Use thematic cases sparingly — they are the fuzziest for `citation_precision` because the gold
set spans multiple products. Keep the anchor list short (2–3 products) so it stays meaningful.

## Distribution checklist for a full set (~30–40 cases)
- Spread anchors across categories present in the corpus (Sports & Outdoors, Amazon Fashion,
  camping, fishing, fitness, pet, automotive…). `products --top 30` shows what's available.
- Mix question intents: `product_feature`, `product_issue`, `comparison`/`theme`, sizing/fit,
  durability, value-for-money.
- Every case must pass `golden_set_tools.py validate` with 0 errors before shipping.
```
