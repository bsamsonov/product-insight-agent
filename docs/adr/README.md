# Architecture Decision Records

Documented architecture decisions for the project. Each ADR describes the context, the alternatives considered, and the decision taken.

- [ADR-0001 — Agent Framework: LangGraph](0001-agent-framework-langgraph.md) — LangGraph as the agent framework
- [ADR-0002 — Vector Database: Qdrant](0002-vector-db-qdrant.md) — Qdrant as the vector database
- [ADR-0003 — Recursive token-based chunking](0003-chunking-strategy.md) — text chunking strategy
- [ADR-0004 — Custom LLM-as-judge](0004-evals-custom-llm-judge.md) — custom metric instead of RAGAS
- [ADR-0005 — Per-tenant Qdrant collections](0005-tenant-isolation-strategy.md) — tenant isolation via collections
- [ADR-0006 — Redis INCRBYFLOAT budget tracking](0006-budget-caps-redis.md) — budget caps via Redis
- [ADR-0007 — RAG over fine-tuning](0007-no-fine-tuning.md) — why we don't fine-tune
- [ADR-0008 — Multi-Provider LLM Strategy](0008-multi-provider-llm-strategy.md) — free-tier first, multi-provider
