"""Test-wide defaults.

The API builds its retriever at startup; keep tests fast and offline by skipping the
BGE-M3 dense index and the cross-encoder reranker (BM25 still runs on the corpus).
"""

import os

os.environ.setdefault("POC_DENSE_RETRIEVAL", "0")
os.environ.setdefault("POC_RERANKER", "0")

# No real backoff sleeps in unit tests (retry logic still runs).
os.environ.setdefault("POC_LLM_RETRY_MAX_WAIT_S", "0")
