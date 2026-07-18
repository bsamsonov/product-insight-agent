# ADR-0001 — Agent Framework: LangGraph

> Status: Accepted
> Date: 2026-05-08
> Deciders: Boris Samsonov (architect)

---

## Context

The Product Insight Agent requires a production-grade framework for orchestrating a multi-step agentic workflow: intent classification → plan → retrieve → cluster → summarize → judge, with optional human-in-the-loop (HITL) checkpoints when the model signals low confidence.

The framework must satisfy several requirements simultaneously:

- **Stateful multi-step flows.** Each request moves through 5–7 nodes with shared state (intent, retrieved chunks, clusters, draft answer, judge verdict). State must be inspectable for debugging and resumable for HITL.
- **Conditional routing.** After the judge node, the graph must branch: confident answer → return, low-confidence → HITL pause or escalate. This is a directed graph, not a linear pipeline.
- **Streaming.** The API must deliver partial results (SSE) as nodes complete, not wait for the full pipeline. LangGraph exposes `astream_events` for this purpose.
- **Checkpointing.** When a human reviewer needs to intervene (HITL), the agent state must be persisted and later resumed. This is non-trivial to implement from scratch.
- **Testability.** Individual nodes must be unit-testable with mocked LLM providers. The framework must not hard-wire provider calls.
- **Production trajectory.** The POC should evolve into a real system. The framework should not impose a ceiling that forces a rewrite later.

Two frameworks were evaluated as primary candidates: **LangGraph** (by LangChain) and **CrewAI**.

---

## Decision

We will use **LangGraph** (v0.2+) as the agent orchestration framework.

LangGraph models the agent as a `StateGraph`: nodes are pure functions `(state) -> partial_state`, edges can be conditional, and checkpointers (in-memory or Postgres-backed) persist the full graph state between steps. This aligns directly with our requirements.

The agent state is defined as a `TypedDict` (`AgentState`) and stored in `packages/agent/src/poc/agent/state.py`. Each node is a separate module under `packages/agent/src/poc/agent/nodes/`. The graph is assembled in `packages/agent/src/poc/agent/graph.py` and compiled once at startup.

For the POC, we use the in-memory checkpointer; upgrading to `AsyncPostgresSaver` for persistent HITL requires changing one line.

---

## Consequences

### Positive consequences

- **Native checkpointing.** Pause-and-resume for HITL requires no custom implementation — just `interrupt_after=["judge"]` and a checkpointer.
- **`astream_events` API** delivers fine-grained events (node start, LLM token, node end) making streaming SSE straightforward.
- **First-class MCP support** (Model Context Protocol) — LangGraph integrates MCP tools natively, which matters if we add external tool calls (Slack, CRM) in later sprints.
- **Production deployments** can use LangGraph Platform (managed) or self-hosted, both without changing the graph code.
- **Mature ecosystem.** LangGraph ships as part of the LangChain family but has no hard dependency on `langchain-core` in graph logic — individual nodes call `poc.llm.LLMProvider` directly.

### Negative consequences / risks

- **Learning curve.** LangGraph's `StateGraph` + `ReducerDict` semantics are non-obvious compared to a plain async function pipeline. Budget ~2–3 hours to internalize the edge/reducer model.
- **LangChain ecosystem coupling.** Even though we avoid `langchain-core` in business logic, transitive dependency upgrades can break things. Pin minor versions.
- **Overhead for simple flows.** A single-node "direct RAG" call has more ceremony than a plain `async def answer(q)`. For Sprint 1 baseline RAG, we use a simpler `RAGPipeline` class; the LangGraph graph is introduced in Sprint 2.

### Neutral / noteworthy

- LangGraph does not prescribe how nodes call LLMs — it is compatible with our `LLMProvider` Protocol and does not force using `langchain` wrappers.
- If the team later wanted to run nodes on separate workers (fan-out), LangGraph's `Send` primitive supports this without architecture changes.

---

## Alternatives Considered

| Option | Pros | Cons | Reason rejected |
|--------|------|------|-----------------|
| **CrewAI** | Fast to prototype; role/task abstraction feels natural for small teams of agents; good docs for beginners | No native checkpointing; limited conditional routing (linear crew by default); production deployments require proprietary CrewAI cloud; graph control flow is implicit and hard to test | Insufficient control for HITL + streaming + conditional branching in one POC |
| **Plain asyncio pipeline** (no framework) | Zero dependencies; trivially testable; maximum control | Must implement state management, streaming, checkpointing, conditional routing from scratch — 300–500 LOC of boilerplate that LangGraph provides for free | Reinventing solved problems; poor ROI for a POC |
| **AutoGen (Microsoft)** | Strong multi-agent conversation model; good for debate/reflection patterns | Conversation-centric model does not map cleanly to a structured retrieve-cluster-summarize graph; less control over state shape; checkpointing less mature | Mismatch between framework model and our use case |
| **Haystack Pipelines** | Excellent RAG-focused pipeline DSL; good component library | Graph model is less flexible for agentic loops and conditional routing; HITL not first-class | Strong for pure RAG, but Sprint 2+ requires agentic control flow |

---

## References

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph checkpointing guide](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph streaming guide](https://langchain-ai.github.io/langgraph/how-tos/streaming/)
- `docs/planning/06_implementation_plan.md` §2.1 — LangGraph state design
- `docs/planning/05_recommended_poc.md` — architecture overview
