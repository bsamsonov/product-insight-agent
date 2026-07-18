# Retrieval — package notes

- Pipeline: BM25 (top-50) + Dense/Qdrant (top-50) → RRF fusion → Reranker (top-k)
- RRF formula: `1.0 / (k + rank + 1)`, k=60 (hardcoded), summed over both lists
- Qdrant collection name: `reviews__{tenant}` — embedding dimension auto-detected from the Embedder
- `QdrantIndex._stable_id()` hashes chunk_id to int: `sha256(chunk_id) % 2^53` (deterministic across processes — re-indexing overwrites the point; builtin `hash()` is unusable, it is salted by PYTHONHASHSEED)
- `QdrantIndex(recreate=True)` (the `--recreate` flag of `scripts/index.py`) drops the collection before creating it — needed when the payload schema/dataset changes
- BM25Index — in-memory, tokenization via `\w+` (lowercase); `build()` must be called before `search()`
- Payload filters → Qdrant FieldCondition, scalar types only (str/int/float/bool)
- `ScoredChunk` — frozen dataclass: fields `chunk` (Chunk) and `score` (float)
