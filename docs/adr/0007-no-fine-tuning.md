# ADR-0007 — RAG over fine-tuning for product insight domain adaptation

> Status: Accepted
> Date: 2026-05-16
> Deciders: Boris Samsonov (architect)

---

## Context

The Product Insight Agent must produce domain-specific answers about product reviews — understanding terminology like "sole delamination", "stitching quality", "true-to-size fit", or industry-specific complaint patterns. A general-purpose LLM has no knowledge of any specific company's product catalog, review history, or domain vocabulary.

Two architecturally distinct approaches exist for incorporating domain-specific knowledge into an LLM-based system:

**RAG (Retrieval-Augmented Generation):** The model's weights remain unchanged. Domain knowledge is stored in an external index (Qdrant in our case). At inference time, relevant chunks are retrieved and injected into the prompt as context. The model reasons over the provided context.

**Fine-tuning:** The model's weights are updated on a domain-specific training set to bake knowledge and behavioural patterns directly into the model's parameters. At inference time, no retrieval step is needed — the model "remembers" domain facts.

Both approaches are well-established in the field. The decision is not which is better in the abstract, but which is the right trade-off for this POC given our constraints: timeline, data availability, cost, and production maintainability.

Several fine-tuning variants were considered: full fine-tuning (updating all weights), LoRA (Low-Rank Adaptation — updating only a small set of adapter matrices), and prompt-only adaptation (no retrieval, no fine-tuning — relying on the base model with a detailed system prompt).

---

## Decision

We will use **RAG only** for domain adaptation in this POC. Fine-tuning of any kind (full, LoRA, RLHF) will not be performed.

The decision is driven by four compounding constraints:

1. **Insufficient training data.** Fine-tuning a large language model to beat a well-prompted base model requires a minimum of 1,000–5,000 high-quality labeled question-answer pairs covering the target domain, with clear diversity across products, complaint types, and languages. Our current corpus consists of synthetic reviews generated for the POC. Synthetic data introduces distribution shift — a model fine-tuned on synthetic data may perform worse on real reviews than the base model prompted with retrieved context.

2. **Time-to-market.** The full fine-tuning pipeline requires: dataset curation and quality filtering → training run → evaluation → safety evaluation → deployment of a new model version. On cloud infrastructure (Google Vertex AI, OpenAI fine-tuning API), this pipeline takes 1–3 weeks for a single iteration. The POC must deliver a working demo in 7 sprints. RAG retrieval is operational from Sprint 1.

3. **Maintainability.** Fine-tuned models become stale as the review corpus grows and evolves. Every time a new product line is added or review sentiment shifts, the fine-tuned model must be re-trained to reflect the updated domain. With RAG, re-indexing new reviews is sufficient — no model update required.

4. **Cost.** Fine-tuning a Gemini 2.5 Pro class model on a dataset of 5,000 examples costs approximately $500–2,000 on cloud APIs (pricing as of May 2026), plus storage of the fine-tuned model. LoRA fine-tuning on an open-weight model (Llama 3.3 70B) requires a GPU instance with 80GB+ VRAM and costs $100–500 per training run. The POC operates on free-tier API quotas where possible. Fine-tuning would consume the entire budget before the first demo.

Furthermore, empirical evidence from production RAG deployments (cited in the RAGAS paper and multiple MLOps case studies) shows that Gemini 2.5 Pro combined with high-quality retrieval and well-structured prompts reaches faithfulness ≥ 0.85 and answer relevance ≥ 0.80 on domain-specific tasks — matching or exceeding LoRA fine-tuned smaller models on retrieval-grounded QA tasks.

---

## Consequences

### Positive consequences

- **Fast iteration.** Adding new data to the knowledge base requires re-running `scripts/index.py` — no model training, no deployment pipeline, no waiting.
- **Transparent domain knowledge.** Every fact in an answer can be traced to a specific retrieved chunk (citation). With fine-tuning, it is impossible to determine whether the model "learned" a fact from training data or is hallucinating — RAG makes the source explicit.
- **No GPU infrastructure required.** The POC runs entirely on CPU (embedding model) and API calls. This removes a significant operational dependency.
- **Provider flexibility.** Because we use API-based models through `LLMProvider`, we can switch providers (Gemini → DeepSeek → Claude) without retraining. A fine-tuned model is locked to the provider it was trained with.
- **Correct trade-off for a POC.** The goal is to demonstrate the architecture and measure quality metrics, not to achieve maximum performance. RAG reaches the quality targets set in `docs/cost-model.md`.

### Negative consequences / risks

- **All domain knowledge must be in the index.** If a fact about a product is not in the indexed review corpus, the agent cannot answer correctly — it will either hallucinate (caught by the Judge node) or return a "not enough information" response. There is no parametric fallback.
- **Retrieval failures propagate directly.** If the retriever returns irrelevant chunks (a retrieval error), the LLM has no ability to compensate from memory. Fine-tuned models can partially recover from retrieval failures using parametric knowledge. This risk is mitigated by the hybrid BM25+dense retrieval strategy (ADR-0003) and the Judge node's faithfulness check.
- **Long-context cost.** RAG injects 2,400–6,000 tokens of context per request. Fine-tuned models need only the question (200 tokens). At high QPS, the context injection cost is significant — see `docs/cost-model.md`.

### Neutral / noteworthy

- ADR-0007 explicitly defers fine-tuning to a hypothetical Phase 2. The trigger for revisiting fine-tuning would be: (a) real labeled data reaching 5,000+ examples, and (b) RAG quality plateauing below targets despite retrieval tuning.
- LoRA fine-tuning of an open-weight model remains a viable Phase 2 enhancement — it would complement (not replace) RAG by improving the model's domain vocabulary understanding.

---

## Alternatives Considered

| Option | Pros | Cons | Reason rejected |
|--------|------|------|-----------------|
| **Full fine-tuning** (cloud API, e.g., OpenAI/Vertex) | Maximum domain adaptation; no retrieval latency | Requires 1000+ quality examples; $500–2000 per run; 1–3 week pipeline; model becomes stale as data changes; incompatible with multi-provider strategy | Cost, data volume, and maintenance overhead rule it out for POC |
| **LoRA fine-tuning** (open-weight model on GPU) | Cheaper than full fine-tuning; adapter weights are small; can be swapped per tenant | Still requires GPU infrastructure ($100–500/run); synthetic training data causes distribution shift; same staleness problem as full fine-tuning | GPU dependency conflicts with local-first development goal; insufficient training data quality |
| **Prompt-only adaptation** (no RAG, no fine-tuning) | Simplest architecture; no indexing pipeline; no vector store | The model has no knowledge of specific product reviews; answers are generic and ungrounded; hallucination rate is high; fails the faithfulness target | Hallucination risk is unacceptable; cannot cite sources; fails the core product requirement |
| **Hybrid RAG + fine-tuning** | Best of both worlds: parametric domain knowledge + explicit retrieval | Requires both the fine-tuning pipeline and the RAG infrastructure; 2× the maintenance burden; overkill for a POC | Phase 2 consideration; not justified at current data volume and timeline |

---

## References

- `packages/retrieval/` — RAG retrieval implementation (the chosen path)
- `docs/cost-model.md` — cost projections that make fine-tuning unaffordable at POC scale
- ADR-0003 — chunking strategy for the RAG index
- ADR-0004 — eval metrics used to validate that RAG meets quality targets without fine-tuning
- [RAGAS paper](https://arxiv.org/abs/2309.15217) — empirical evidence for RAG quality on domain-specific tasks
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation of Large Language Models
- `docs/planning/06_implementation_plan.md` §B — "Fine-tuning: see ADR-0007"
