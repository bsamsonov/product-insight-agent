#!/usr/bin/env bash
#
# run_main.sh — launch the main Product Insight Agent (full LangGraph pipeline)
#               with OpenTelemetry -> Langfuse tracing.
#
# The agent (poc.agent.ProductInsightAgent) runs the
# intent -> plan -> retrieve -> cluster -> summarize -> judge state machine.
# Tracing is configured by poc.observability.tracing.configure(): when LANGFUSE_HOST
# is set, OTel spans are shipped to ${LANGFUSE_HOST}/api/public/otel.
#
# Two modes:
#   serve  (default) — start the FastAPI app; the agent runs per /ask request and
#                      every node + LLM call is exported as a Langfuse span.
#   eval             — batch-run the LangGraph agent over the golden set.
#
# Usage:
#   ./run_main.sh                       # serve on :8000
#   ./run_main.sh serve --port 9000     # extra args passed to uvicorn
#   ./run_main.sh eval --limit 3        # run the agent over 3 golden-set cases
#   ./run_main.sh eval --limit 5 --no-agent   # baseline RAG instead of the agent
#
# Once serving, query the agent:
#   curl -s localhost:8000/ask -H 'content-type: application/json' \
#     -d '{"question":"What do customers say about battery life?"}' | jq
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Load .env so the LLM provider keys and LANGFUSE_* OTLP export config are present.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODE="${1:-serve}"
[[ $# -gt 0 ]] && shift || true

if [[ -n "${LANGFUSE_HOST:-}" && -n "${LANGFUSE_PUBLIC_KEY:-}" ]]; then
  echo "[run_main] Langfuse OTLP export ON -> ${LANGFUSE_HOST}/api/public/otel"
else
  echo "[run_main] Langfuse OTLP export OFF (LANGFUSE_HOST not set) — local spans only"
fi

case "$MODE" in
  serve)
    PORT="${POC_API_PORT:-8000}"
    # If the caller didn't pass --port, apply our default.
    if [[ "$*" != *"--port"* ]]; then
      set -- --port "$PORT" "$@"
    fi
    echo "[run_main] starting LangGraph agent API: uvicorn api.main:app $*"
    echo "[run_main] POST http://localhost:${PORT}/ask   (docs: /docs)"
    echo
    exec uv run uvicorn api.main:app "$@"
    ;;
  eval)
    # Default to the full LangGraph agent unless the caller overrides with --no-agent.
    if [[ "$*" != *"-agent"* ]]; then
      set -- --agent "$@"
    fi
    echo "[run_main] uv run python scripts/run_eval.py $*"
    echo
    exec uv run python scripts/run_eval.py "$@"
    ;;
  *)
    echo "[run_main] unknown mode '$MODE' (expected: serve | eval)" >&2
    echo "  ./run_main.sh serve [uvicorn args]" >&2
    echo "  ./run_main.sh eval  [run_eval args]" >&2
    exit 2
    ;;
esac
