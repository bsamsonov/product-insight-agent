# ADR-0004 — Custom LLM-as-judge metrics instead of RAGAS library

> Status: Accepted
> Date: 2026-05-16
> Deciders: Boris Samsonov (architect)

---

## Context

The Product Insight Agent requires quantitative evaluation of RAG pipeline quality. Without automated metrics, the only feedback signal is manual inspection of individual answers — which does not scale and cannot be integrated into CI.

The evaluation system must measure four dimensions of RAG quality that are now standard in the field:

- **Faithfulness** — does every claim in the answer appear in the retrieved context? (measures hallucination)
- **Answer Relevance** — does the answer actually address the user's question?
- **Context Precision** — are the retrieved chunks relevant to the question? (retrieval quality)
- **Context Recall** — does the retrieved context contain enough information to answer the question?

**RAGAS** (Retrieval Augmented Generation Assessment) is the dominant open-source library for exactly these metrics. It was the first candidate evaluated. However, RAGAS introduces a dependency on specific LangChain model wrappers and assumes a particular interface for interacting with LLMs that conflicts with our `LLMProvider` abstraction (defined in `packages/llm/`).

The specific compatibility problems observed:

1. RAGAS expects LLM clients to conform to LangChain's `BaseChatModel` interface. Our `LLMProvider` is a simpler `Protocol` that is provider-agnostic.
2. RAGAS version `0.1.x` had breaking changes between minor versions during the evaluation period, causing import errors after routine `uv sync`.
3. RAGAS instantiates its own OpenAI client internally, bypassing our `LLMRouter` which handles provider fallback, rate-limit retry, and budget tracking.

The alternative approaches considered were: RAGAS (with compatibility shims), pure human evaluation, exact-match / keyword-based metrics, and a custom LLM-as-judge implementation.

---

## Decision

We will implement **custom LLM-as-judge metrics** that follow the RAGAS methodology but use our own `LLMProvider` abstraction for all model calls.

The implementation lives in `packages/evals/src/poc/evals/metrics_baseline.py`. Each metric is a separate async function that:

1. Constructs a structured evaluation prompt (loaded from `packages/prompts/data/`)
2. Calls `LLMProvider.complete()` requesting a JSON-structured response
3. Parses the score (0.0–1.0) from the response using Pydantic

The metrics are compatible in semantics with RAGAS: scores produced by our implementation should be within ±5% of RAGAS scores on the same inputs (validated manually on the golden set). This allows us to cite RAGAS benchmarks as approximate targets.

The eval runner in `packages/evals/src/poc/evals/runner.py` accepts a golden set JSONL file (question, reference answer, expected context) and produces a metrics report. It is invoked via `scripts/run_eval.py`.

---

## Consequences

### Positive consequences

- **Full `LLMProvider` compatibility.** The eval metrics use the same provider abstraction as the agent — including model routing, prompt caching, and budget tracking. Eval costs appear in the same billing view as production costs.
- **Prompt control.** We can inspect and modify the judge prompts directly. If a metric consistently over- or under-estimates quality on our domain (product reviews), we can tune the judge prompt without waiting for a RAGAS release.
- **No LangChain dependency in evals.** The `poc-evals` package has no dependency on `langchain-core`, reducing the risk of transitive version conflicts.
- **Consistent structured output.** We use Pydantic to parse judge responses, so malformed LLM outputs fail loudly (a `ValidationError`) rather than silently producing a default score.

### Negative consequences / risks

- **LLM cost per eval run.** Each evaluated question requires 2 LLM calls (one for faithfulness/groundedness, one for answer relevance). On Gemini Flash free tier this is free up to ~750 evaluations per day; on paid tier it costs approximately $0.001 per question at 3200-token inputs. A golden set of 50 questions costs ~$0.05 to evaluate — acceptable.
- **Non-determinism.** LLM judge scores vary by ±3% across repeated runs at temperature=0 (due to floating-point differences in sampling). CI thresholds must account for this variance. We set the CI gate at `faithfulness ≥ 0.82` (3% below the 0.85 target) to avoid flaky failures.
- **Maintenance burden.** We now own the judge prompts. If the scoring approach needs updating (e.g., new metric added to the field), we implement it ourselves rather than getting it from a RAGAS release.
- **Not a drop-in RAGAS replacement.** Downstream tooling that expects RAGAS score schemas will not work without adaptation.

### Neutral / noteworthy

- The four RAGAS metrics (faithfulness, answer_relevance, context_precision, context_recall) are well-documented in the [RAGAS paper](https://arxiv.org/abs/2309.15217). Our prompts re-implement the same decomposed approach.
- If RAGAS resolves the LangChain coupling in a future release, we can re-evaluate adopting it without changing the metric interface — just swap the implementation behind the same function signatures.

---

## Alternatives Considered

| Option | Pros | Cons | Reason rejected |
|--------|------|------|-----------------|
| **RAGAS directly** (with compatibility shim) | Battle-tested; community maintained; direct comparison with published benchmarks | Required wrapping `LLMProvider` in a `BaseChatModel` shim — fragile adapter that broke on every RAGAS minor version update; bypasses our LLMRouter budget tracking | Version instability + broken budget tracking were blocking issues |
| **Human evaluation only** | Highest accuracy; catches nuanced issues | Not automatable; cannot run in CI; too slow and expensive for iterative development (each eval cycle would require manual review of 50+ answers) | No CI integration possible |
| **Exact-match / keyword metrics** (e.g., BLEU, ROUGE) | Zero LLM cost; fully deterministic | Penalizes correct paraphrases; rewards keyword stuffing; cannot measure faithfulness (a fluent hallucination scores high on ROUGE); not suitable for open-ended generative answers | Does not capture semantic correctness |
| **G-Eval / custom rubric (non-RAGAS)** | Freedom to define any rubric | No baseline to compare against; harder to communicate quality to stakeholders familiar with RAGAS | RAGAS methodology is the field standard; deviating makes benchmarking harder |

---

## References

- `packages/evals/src/poc/evals/metrics_baseline.py` — implementation
- `packages/evals/src/poc/evals/runner.py` — eval runner
- `packages/evals/data/` — golden set JSONL files
- [RAGAS paper](https://arxiv.org/abs/2309.15217) — ES Ragas: Automated Evaluation of Retrieval Augmented Generation
- [RAGAS documentation](https://docs.ragas.io/)
- `docs/cost-model.md` — eval cost estimates
