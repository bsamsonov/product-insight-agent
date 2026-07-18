# ADR-0008 — Multi-Provider LLM Strategy (Free-Tier First)

> Status: Accepted
> Date: 2026-05-08
> Deciders: Boris Samsonov (architect)
> Supersedes: initial assumption of Anthropic Claude as the sole provider

---

## Context

The Product Insight Agent was originally designed with Anthropic (Claude Haiku / Sonnet / Opus) as the LLM backend. However, for a solo-developer POC aimed at building and demonstrating AI architecture skills, the ongoing API cost of Anthropic is prohibitive (~$20–50/month even for moderate development activity).

More importantly, the architecture already separates provider concerns behind a `LLMProvider` Protocol — no business logic calls Anthropic directly. This makes provider substitution a configuration concern, not an implementation change.

The decision space centers on three tensions:

1. **Cost vs. quality.** Free tiers impose rate limits. A model that is slow or unavailable degrades the development loop. However, for a POC, "good enough" quality at zero cost is better than "best quality" at $30/month.
2. **OpenAI-compatible API vs. native SDKs.** Maintaining three separate SDK integrations (Anthropic, OpenAI, Google) inflates complexity. A unified approach reduces the `llm` package to one class.
3. **Privacy and PII.** Free tiers of cloud APIs typically log prompts for model improvement. For reviews that may contain user PII, this is a risk that must be acknowledged and mitigated (redact before sending, or use local models).

The key technical insight: **Gemini AI Studio, Groq, and DeepSeek all expose OpenAI-compatible REST endpoints.** A single `AsyncOpenAI` client parameterized by `base_url` and `api_key` can reach all three providers — no separate SDK per provider.

---

## Decision

We will use a **multi-provider, free-tier-first stack** built entirely on the OpenAI-compatible API surface:

| Role | Primary (free) | Fallback | Cost at overflow |
|------|---------------|----------|-----------------|
| **fast** (intent, judge, classifiers) | Gemini 2.5 Flash — AI Studio | Groq Llama 3.3 70B | $0.075 / 1M input tokens |
| **smart** (plan, summarize, RAG answer) | Gemini 2.5 Pro — AI Studio | DeepSeek V3.1 | $1.25 / 1M (Gemini) or $0.27/1M (DeepSeek) |
| **escalate** (low-confidence reasoning) | DeepSeek-R1 | Gemini 2.5 Pro Thinking | $0.55 / 1M input tokens |
| **embeddings** | local `bge-m3` (CPU) | Gemini text-embedding-004 | free at 1500 RPM |
| **reranker** | local `bge-reranker-v2-m3` | Jina Reranker v2 | 1M req/month free |

The implementation uses a **single adapter class** `OpenAICompatibleProvider` (`packages/llm/src/poc/llm/openai_compatible.py`) configured by `provider_name`, `base_url`, and `api_key`. A `registry.py` module holds default `base_url` values for known providers so callers only specify `provider_name`.

Provider routing (which model/provider for which task) is declared in `packages/llm/data/router.yaml` with `primary` / `fallback` pairs per logical role. The model router (Sprint 3) reads this configuration and enforces budget caps.

Anthropic remains available as an optional second adapter (`anthropic_provider.py`, marked `# pragma: optional`) for benchmarking purposes. It is not activated in the default stack.

---

## Consequences

### Positive consequences

- **Zero LLM cost during development.** The Gemini AI Studio free tier (1500 RPD for Flash, 25 RPD for Pro) covers typical development activity. Groq is the fast fallback when Gemini rate-limits are hit. Total expected spend: $0–5/month.
- **Simpler codebase.** One adapter class (`OpenAICompatibleProvider`) instead of one per provider. Adding a new provider (Mistral, Together, Cerebras) requires adding one entry to `registry.yaml` — no new Python code.
- **Architecture demonstration value.** The pattern "one interface, multiple providers via configuration" is a stronger portfolio signal than "we used Claude." It demonstrates that the candidate understands provider abstraction, cost engineering, and multi-vendor strategy.
- **Implicit prompt caching on Gemini.** Requests with stable prefixes ≥1024 tokens automatically benefit from 75% discount on repeated tokens. This is relevant for the RAG prompt (system message + retrieved context stays stable across similar queries).
- **Offline / air-gapped development.** Ollama (local) is a first-class provider in the registry. Any `llama3.2`, `qwen2.5`, or `deepseek-r1-distill` model available locally can be used without internet access.

### Negative consequences / risks

- **Rate limit management.** Free tiers impose hard per-minute and per-day limits (Gemini Pro: 5 RPM / 25 RPD). The fallback chain mitigates this but adds complexity to the router (Sprint 3). During evaluation runs on the 30-question golden set, Pro calls may exhaust the daily quota.
- **Privacy: Gemini AI Studio logs prompts.** Google's AI Studio free tier terms state that interactions may be used to improve models. For reviews containing PII (names, emails), the ingestion PII-redaction step (Sprint 4) becomes a hard prerequisite before using cloud LLMs. Until Sprint 4 is done, use local Ollama or ensure test data is synthetic.
- **Model parity is not guaranteed.** Gemini 2.5 Flash and Groq Llama 3.3 do not behave identically. Prompt engineering for structured output must be tested against both. JSON schema validation with retry on invalid output (already in `OpenAICompatibleProvider`) mitigates this.
- **DeepSeek PII concern.** DeepSeek's servers are hosted in China. For production use with real customer data, this is a compliance risk. In POC, use only synthetic data with DeepSeek.

### Neutral / noteworthy

- The `LLMProvider` Protocol signature is identical regardless of provider. Switching the entire stack back to Anthropic for a production deployment requires changing `router.yaml` and setting `ANTHROPIC_API_KEY` — one file, one env var.
- Pricing is tracked in `packages/llm/data/pricing.yaml`. Free-tier limits are annotated with `free_tier_rpd` / `free_tier_rpm` fields so the cost estimator can correctly return `$0.00` for requests within quota.

---

## Alternatives Considered

| Option | Pros | Cons | Reason rejected |
|--------|------|------|-----------------|
| **Anthropic Claude only** (original plan) | Best-in-class instruction following; native prompt caching; predictable behavior; strong structured output | ~$20–50/month at development intensity; single vendor dependency; Anthropic Python SDK adds a second non-OpenAI-compatible code path | Cost and single-vendor risk are unacceptable for a solo POC |
| **OpenAI GPT-4o / GPT-4o-mini** | Excellent structured output; well-documented; broad community | No free tier (only $5 trial credit); GPT-4o is expensive at heavy use; same single-vendor risk | No sustained free tier; not meaningfully different from Anthropic for our use case |
| **Single local model (Ollama Llama 3.3 70B)** | Completely free; zero PII risk; works offline | Llama 70B requires ≥48 GB VRAM for good performance; CPU inference is 5–15 tok/s — too slow for development feedback loop; quality on structured tasks is lower than Gemini Pro | Insufficient quality and speed for a development loop; fine as a fallback / offline option |
| **OpenRouter unified API** | One API key for all models; free-tier models available; easy swapping | Free-tier models on OpenRouter are unstable (may be disabled); extra latency hop; slightly higher pricing than direct APIs | Instability and latency overhead unjustified when direct APIs are trivially configurable |
| **Mistral La Plateforme** | European hosting (GDPR-friendly); good quality for size; open-source models | Free tier is very restrictive (1 RPS "experimental"); Pro models are paid; smaller English benchmark scores | Too restrictive for development use; better as a future GDPR-compliance option |

---

## References

- `docs/planning/07_free_tier_strategy.md` — detailed provider comparison table and router YAML spec (primary source)
- [Google AI Studio free tier limits](https://ai.google.dev/pricing)
- [Groq free tier documentation](https://console.groq.com/docs/rate-limits)
- [DeepSeek API pricing](https://platform.deepseek.com/docs/pricing)
- `packages/llm/data/pricing.yaml` — machine-readable pricing table used by `poc.llm.pricing`
- `packages/llm/src/poc/llm/registry.py` — provider registry with default `base_url` values
- ADR-0001 — agent framework (uses the same `LLMProvider` abstraction)
