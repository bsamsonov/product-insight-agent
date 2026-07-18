# Cost Model — Product Insight Agent

> Last updated: 2026-05-16
> Pricing basis: May 2026 public API pricing (approximate; verify before production budget planning)

---

## 1. Assumptions

### Per-request token budget

Each user request triggers the following LLM calls through the LangGraph agent:

| Call | Role | Input tokens | Output tokens |
|------|------|-------------|---------------|
| Intent Classifier | Classify question type and extract filters | 600 (system prompt) + 200 (question) = **800** | **50** |
| Plan + Summarizer | Generate structured answer from retrieved context | 600 (system prompt) + 2400 (retrieved chunks) + 200 (question) = **3200** | **400** |
| Quality Judge | Evaluate faithfulness and answer relevance | 600 (system prompt) + 2400 (context) + 400 (draft answer) = **3400** | **100** |

**Total per user request: ~7,400 input tokens, ~550 output tokens across 3 LLM calls.**

For eval runs, the judge runs 2 additional LLM calls per evaluated question (faithfulness + answer relevance), adding ~7,000 input tokens per eval question.

### Prompt caching

For providers that support prompt caching (Gemini, Anthropic), the 600-token system prompt is cached after the first call. Subsequent calls in the same session save ~600 input tokens per call.

- With caching active: effective input tokens per request ≈ 7,400 - (3 × 600 × 0.75) = 7,400 - 1,350 = **~6,050 tokens** (18% savings)
- Cache hit assumed at 75% of requests in steady state (new session = no cache)

---

## 2. Provider Pricing Table

Pricing as of May 2026. All prices in USD per 1 million tokens unless noted.

| Provider / Model | Input ($/1M) | Output ($/1M) | Cache Hit Input ($/1M) | Free Tier |
|---|---|---|---|---|
| **Gemini 2.5 Flash** | $0.075 | $0.30 | $0.019 | 1,500 RPD (requests/day) |
| **Gemini 2.5 Pro** | $1.25 | $10.00 | $0.31 | 50 RPD |
| **Groq Llama 3.3 70B** | $0.59 | $0.79 | n/a | ~14,400 RPD (generous free tier) |
| **DeepSeek V3** | $0.27 | $1.10 | $0.07 | No free tier |
| **Anthropic Claude Haiku 4.5** | $0.80 | $4.00 | $0.08 | No free tier |

Notes:
- Gemini Flash free tier: 1,500 RPD at 15 RPM (requests per minute)
- Gemini Pro free tier: 50 RPD at 2 RPM — effectively demo/test only
- Groq free tier: varies by model; Llama 3.3 70B is approximately 14,400 RPD at ~100 RPM
- DeepSeek cache discount applies only to the prefix matching a previously seen prompt

---

## 3. Per-Request Cost Estimate

### Routing strategy (as configured in ADR-0008)

The model router assigns different models to different roles:
- **Classifier + Judge:** Gemini 2.5 Flash (fast, cheap, sufficient for classification)
- **Summarizer:** Gemini 2.5 Pro (best output quality for the user-facing answer)

### Free tier scenario (Gemini Flash for all calls + Groq fallback)

Within free tier limits, cost = **$0.00 per request**.

Free tier capacity:
- Gemini 2.5 Flash: 1,500 RPD → viable up to **~1 RPM sustained** (with bursts to 15 RPM)
- Groq Llama 3.3 70B: ~14,400 RPD → viable up to **~10 RPM sustained**

### Paid tier — mixed routing (Flash for classifier + judge, Pro for summarizer)

```
Classifier (Gemini Flash):
  Input:  800 tokens × $0.075/1M  = $0.00006
  Output:  50 tokens × $0.300/1M  = $0.000015

Summarizer (Gemini Pro):
  Input:  3200 tokens × $1.25/1M  = $0.0040
  Output:  400 tokens × $10.00/1M = $0.0040

Judge (Gemini Flash):
  Input:  3400 tokens × $0.075/1M  = $0.000255
  Output:  100 tokens × $0.300/1M  = $0.00003

Total per request (no cache): ~$0.0083
```

**Rounded estimate: ~$0.008–$0.012 per request** depending on context length variation.

### With prompt caching (75% cache hit on system prompts)

Effective input savings: ~$0.001 per request (3 calls × 600 tokens × 75% × $1.25/1M for Pro, $0.075/1M for Flash).

**With caching: ~$0.007–$0.010 per request.**

---

## 4. Scale Projections

| QPS | Daily Requests | Free Tier Viable? | Cost/Day (paid, mixed routing) | Cost/Month (paid) |
|-----|---------------|-------------------|---------------------------------|--------------------|
| 0.01 | 864 | Yes (Gemini Flash) | $0 | $0 |
| 0.1 | 8,640 | Yes (Groq) | $0 | $0 |
| 0.5 | 43,200 | No | ~$360 | ~$10,800 |
| 1 | 86,400 | No | ~$720 | ~$21,600 |
| 10 | 864,000 | No | ~$7,200 | ~$216,000 |
| 100 | 8,640,000 | No | ~$72,000 | ~$2,160,000 |

Notes:
- "Free Tier Viable" means daily requests fit within the largest available free tier (Groq at ~14,400 RPD)
- Paid costs use the mixed routing strategy ($0.0083/request baseline)
- No caching discount applied to projections (conservative estimate)
- At 10+ QPS, provider rate limits become the binding constraint before cost

### Key thresholds

- **POC demo (0.01–0.1 QPS):** Fully covered by free tiers. Cost = $0.
- **Internal beta (0.1–0.5 QPS):** Groq handles the load for free, but Gemini Pro quality may be desired. Estimated cost: $50–$360/day.
- **Production (1+ QPS):** Paid tier required. Dominant cost is Gemini Pro for the summarizer node.

---

## 5. Break-Even Analysis

### Free tier break-even

| Provider | Free RPD | Max viable QPS (sustained) |
|---|---|---|
| Gemini 2.5 Flash | 1,500 | ~0.017 QPS |
| Gemini 2.5 Pro | 50 | ~0.0006 QPS (demo only) |
| Groq Llama 3.3 70B | ~14,400 | ~0.17 QPS |

**Practical free-tier ceiling: ~10 RPM burst, ~0.1 QPS sustained using Groq for all calls.**

At this scale, the POC is entirely free. This covers development, demos, and small-scale testing (up to ~10,000 requests/day).

### When to move to paid

The trigger to switch to paid tier is when daily requests consistently exceed 14,400 (Groq ceiling) or when Gemini Pro quality is required (Groq does not match Gemini 2.5 Pro on complex multi-hop reasoning over long review contexts).

Minimum monthly spend on paid tier at 1 QPS = **~$21,600/month** using the current routing strategy.

---

## 6. Optimization Strategies

In descending order of impact:

1. **Prompt caching (18–30% savings).** Enable Gemini prompt caching on the system prompt and retrieved context prefix. Most effective when the same user asks follow-up questions in the same session (cache TTL = 5 minutes on Gemini). Expected saving: $0.001–$0.003 per request.

2. **Gemini Flash vs Pro routing (10× cost difference).** Use Gemini Pro only for the user-facing Summarizer node. Classifier and Judge use Gemini Flash. The router in ADR-0008 implements this. Switching the Summarizer from Pro to Flash cuts per-request cost from ~$0.0083 to ~$0.0003 — a 27× reduction — at an estimated 15–20% quality drop on complex queries.

3. **BM25-only path for simple factual queries.** For intent `= FACTUAL_SIMPLE`, the classifier can route directly to BM25 retrieval + a single Flash call (no Pro). Expected cost: ~$0.0001/request. Applicable to ~20% of real-world queries based on the golden set distribution.

4. **Response caching in Redis (repeat queries = $0).** Cache the final `Answer` object keyed by `(tenant_id, question_hash, top_k_chunk_hashes)`. Cache TTL = 1 hour. For a product insights use case where users often ask the same category questions (e.g., "what are the top complaints about comfort?"), cache hit rates of 30–50% are realistic during business hours. Expected average saving: 30–50% of requests.

5. **Shorter context windows.** Reducing retrieved context from 2,400 tokens (6 chunks × 400 tokens) to 1,600 tokens (4 chunks) saves ~$0.001/request on Pro calls with minimal quality impact for single-topic queries (multi-topic queries may regress).

6. **Batch eval with off-peak scheduling.** Run nightly eval jobs during off-peak hours when rate limits are less constrained. This does not save money but improves throughput and avoids rate-limit errors during business hours.

---

## 7. Cost Model Assumptions and Caveats

- Prices are approximate and subject to change. Always verify against the provider's pricing page before committing to a budget.
- Token counts are based on the POC's current prompt templates. Prompt engineering changes will shift these numbers.
- The "3 LLM calls per request" assumption holds for the happy path. Retry logic (max 2 retries on provider errors) can triple the cost in worst-case scenarios.
- Eval costs are not included in the per-request model — they are a separate operational cost of approximately $0.05 per 50-question golden set evaluation run.
- Embedding costs (BGE-M3 runs locally) are excluded from this model. Cloud embedding (e.g., text-embedding-3-small at $0.020/1M tokens) would add ~$0.000064/request for a 3,200-token query+context embedding.
