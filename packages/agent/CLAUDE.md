# Agent — package notes

- `AgentState` — TypedDict, not a dataclass: initialize via the `initial_state()` helper
- Node order: `intent → plan → retrieve → cluster → summarize → judge → hitl(conditional) → END`
- Judge node takes `retrieved[:8]` — at most 8 chunks for faithfulness evaluation
- HITL triggers when score < 0.6 (constant `_JUDGE_THRESHOLD` in nodes/judge.py)
- Citations in the draft must use the `[chunk_id]` format — regex `\[([^\]]+)\]` in the judge node
- `Answer` is assembled in the judge node together with the total cost of all nodes
- Traces — a list of dicts; each node appends its record manually via `append`
