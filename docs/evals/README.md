# Evaluation results

Offline evaluation of the full LangGraph agent on the real corpus. The raw per-case
reports are the JSON files in this folder; the table below is generated from them with
`scripts/eval_table.py`, not typed by hand.

## Setup

| | |
|---|---|
| Corpus | Amazon Reviews 2023, `Sports_and_Outdoors`: 12,000 reviews, 558 products, 12,056 chunks |
| Golden set | [`golden_set_v2.jsonl`](../../packages/evals/data/golden_set_v2.jsonl): 35 cases, each anchored to one product; expected substrings and chunk ids are checked against the corpus by `scripts/golden_set_tools.py validate` |
| Pipeline | `ProductInsightAgent`: intent → plan → retrieve → cluster → summarize → judge → groundedness |
| Retrieval | **Hybrid**: BM25 + BGE-M3 dense (Qdrant) fused with RRF, cross-encoder rerank. **BM25 only**: same pipeline with Qdrant switched off |
| Model | `gemini-3.1-flash-lite` for every node and for the faithfulness judge (free tier) |
| Pacing | `POC_GEMINI_RPM=12`; pacing waits are subtracted from latency |

Reproduce:

```bash
uv run python scripts/download_dataset.py
uv run python scripts/index.py --source data/raw/reviews.jsonl --recreate
POC_GEMINI_RPM=12 uv run python scripts/run_eval.py --golden-set golden_set_v2 \
  --data-path data/raw/reviews.jsonl --agent --model gemini-3.1-flash-lite \
  --metrics citation_precision,groundedness_heuristic,faithfulness \
  --label Hybrid --output docs/evals/<date>_golden_set_v2_agent_hybrid.json
# add --qdrant-url http://127.0.0.1:1 --label "BM25 only" for the ablation
uv run python scripts/eval_table.py docs/evals/*_agent_hybrid.json docs/evals/*_agent_bm25only.json
```

Every case is also a Langfuse trace (`eval.golden_set_v2.<case>`) with the metrics
attached as scores, grouped into one session per run (see [../observability.md](../observability.md)).

## Metrics

| Metric | What it measures |
|---|---|
| Faithfulness | LLM judge score (0–1): are the answer's claims supported by the chunks the answer actually used |
| Citation precision | Share of cited chunk ids that belong to the anchor product (`expected_chunk_ids`) |
| Sentences with citations | `groundedness_heuristic`: share of answer sentences (> 20 chars) carrying a `[chunk]` marker; no LLM |
| Expected-substring match | Answer contains the case's expected content words |
| Latency | Wall-clock answer time (retrieval + all LLM calls), metrics and pacing excluded |
| Cost per answer | Estimated from token usage at paid-tier prices; $0 on the free tier |

## Results

_Pending: the hybrid run is re-executed after the fixes below; see the note at the end._

## What the eval runs caught

Running the full eval against the real corpus, with Langfuse traces for every case,
surfaced a series of bugs that unit tests and the keyless CI gate did not:

1. **Structured output silently failed on Gemini.** The OpenAI-compatible endpoint rejects
   the native `response_mime_type` field with HTTP 400, so intent, plan and judge fell back
   to defaults and the judge always reported 0.5.
2. **BM25 indexed only the first 2,000 of 12,000 reviews** in `run_eval.py`, hiding most
   golden-set products from the sparse retriever.
3. **Grouped citations were dropped.** `[a, b, c]` was parsed as one id, which emptied
   `Answer.citations` and depressed citation precision.
4. **Faithfulness was judged against the wrong context**: a fresh unfiltered top-5 search
   instead of the chunks the answer used.
5. **Numeric plan filters crashed hybrid retrieval.** `rating: 2.0` is not a valid Qdrant
   `MatchValue`; the error also discarded the BM25 hits, so every complaint question
   (6 of 35) answered `UNKNOWN` in hybrid mode.
6. **Free-tier quotas.** Requests-per-minute limits are now paced client-side, per-day
   quota errors are not retried, failed metric calls are excluded from the means, and a
   run aborts after three dead cases instead of writing a report of fallback answers.

## Known limitations

- 35 cases is small; per-metric means move by several points between runs.
- `answer_relevance` scored 1.00 on every case in an earlier run, so it does not
  discriminate on this set and is left out of the headline table.
- `expected_chunk_ids` list all chunks of the anchor product, so citation precision
  measures "cites the right product", not "cites the best chunk".
