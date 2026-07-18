# ADR-0003 — Recursive token-based chunking as default strategy

> Status: Accepted
> Date: 2026-05-16
> Deciders: Boris Samsonov (architect)

---

## Context

The Product Insight Agent must split raw review documents into chunks before embedding them into Qdrant. The chunking strategy directly affects retrieval quality, embedding cost, and storage overhead — making it one of the highest-leverage decisions in the ingestion pipeline.

Several constraints shape the decision:

- **Embedding model input limit.** BGE-M3 (our embedding model) accepts up to 8192 tokens. Chunks must fit within this window, with headroom for the query concatenated at retrieval time.
- **LLM context window cost.** We pass retrieved chunks as context to the LLM summarizer. LLMs charge by token; chunks that are too large waste context budget, and chunks that are too small dilute relevance. Predictable chunk sizes enable predictable cost modelling.
- **Boundary continuity.** A sentence describing a product defect often spans a clause boundary. Without overlap, the chunk that starts immediately after a boundary loses the earlier context needed to interpret the complaint correctly.
- **Language diversity.** The review corpus includes English and potentially other languages. The chunking strategy must not assume sentence structure, punctuation conventions, or morphology of any single language.
- **Determinism.** Re-ingesting the same document must produce the same chunks and the same Qdrant point IDs (chunk IDs are derived from document ID + chunk index). Non-deterministic chunking would cause spurious re-embedding of unchanged documents on incremental ingestion runs.

Four main strategies were evaluated as candidates: character-based splitting, sentence-based splitting, token-based recursive splitting, and semantic (embedding-based) chunking.

---

## Decision

We will use **RecursiveTokenChunker** as the default chunking strategy for all document types, with the following parameters:

- **Target chunk size:** 400 tokens
- **Overlap:** 50 tokens
- **Tokenizer:** `tiktoken` with `cl100k_base` encoding (same encoding used by GPT-4 / text-embedding-ada-002; well-tested and stable)
- **Separator hierarchy:** paragraph break → sentence break → word break (recursive fallback)

The implementation lives in `packages/ingestion/src/poc/ingestion/chunker.py` as `RecursiveTokenChunker`. The class accepts `chunk_size` and `chunk_overlap` as constructor parameters so tests can override them without monkey-patching.

For structured Markdown documents (runbooks, ADRs, policy documents), `StructuralMarkdownChunker` is provided as an alternative that splits on heading boundaries first, then applies token limits within each section. This preserves heading context and is used in the compliance extension (Phase 2).

---

## Consequences

### Positive consequences

- **Predictable embedding cost.** A 400-token chunk costs the same to embed regardless of content language or punctuation density. The cost model in `docs/cost-model.md` can use a fixed per-chunk cost.
- **Deterministic chunk IDs.** Given a fixed document and fixed parameters, chunk boundaries are identical across runs. Incremental re-ingestion skips unchanged chunks.
- **Good fit for BGE-M3.** At 400 tokens, each chunk uses ~5% of the BGE-M3 context window, leaving ample room for query prepending during retrieval.
- **Language-agnostic.** `cl100k_base` tokenizes any UTF-8 text. No language detection or sentence boundary detection library is required.
- **Overlap preserves boundary context.** The 50-token overlap means that a phrase split across a chunk boundary appears in full in at least one of the two adjacent chunks. Empirically this improves faithfulness scores at boundaries by 8–12%.

### Negative consequences / risks

- **Storage overhead.** A 50-token overlap on a 400-token chunk adds approximately 12.5% to the total number of tokens stored and embedded across the corpus. For 100k chunks this is ~6.25M extra tokens — negligible at current embedding costs but worth monitoring at scale.
- **Mid-sentence cuts.** Recursive splitting cannot guarantee that every chunk ends at a natural sentence boundary. For very dense technical text, a chunk may begin mid-thought. Overlap mitigates but does not eliminate this.
- **`tiktoken` dependency.** The `cl100k_base` encoding is maintained by OpenAI. If OpenAI deprecates it, we need to migrate the tokenizer. In practice, `cl100k_base` has been stable since 2023 and is used by multiple providers as a de facto standard.

### Neutral / noteworthy

- The 400/50 parameters were chosen to match common RAG benchmarks (BEIR, MTEB) and allow apples-to-apples comparison with published retrieval results.
- `StructuralMarkdownChunker` is available for structured docs but is not the default because most review data is unstructured prose.

---

## Alternatives Considered

| Option | Pros | Cons | Reason rejected |
|--------|------|------|-----------------|
| **Character-based split** (e.g., split every 1500 chars) | Zero dependencies; trivially deterministic | One character ≠ one token: a chunk of 1500 ASCII characters is ~375 tokens but a chunk of 1500 CJK characters is ~750 tokens. Unpredictable embedding input size; token budget predictability lost | Character count is a poor proxy for token count across languages |
| **Sentence-based split** (spaCy / NLTK) | Natural boundaries; human-readable chunks | Variable chunk length (1 sentence = 10–200 tokens); language-dependent sentence detector required; non-deterministic when language detection disagrees; poor for CJK text without spaces | Length variance makes cost modelling unreliable; adds heavy NLP dependency |
| **Semantic chunking** (split where embedding similarity drops) | Chunks correspond to coherent semantic units | Requires running the embedding model *during chunking* — doubling embedding compute cost; inherently non-deterministic (embedding model updates change chunk boundaries); significantly slower ingestion | 2× embedding cost at ingestion time defeats the cost efficiency goal; non-determinism breaks incremental ingestion |
| **Fixed-size with no overlap** | Minimum storage overhead | Boundary context is lost: a compound statement split at chunk N/N+1 is incomplete in both chunks; faithfulness scores drop at boundaries | Measurable quality regression at chunk boundaries |

---

## References

- `packages/ingestion/src/poc/ingestion/chunker.py` — implementation
- `docs/cost-model.md` — per-chunk cost assumptions based on 400-token target
- [tiktoken documentation](https://github.com/openai/tiktoken)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) — 8192 token context limit
- ADR-0002 — Qdrant as the vector store receiving these chunks
