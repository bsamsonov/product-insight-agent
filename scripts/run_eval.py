"""CLI to run baseline RAG evaluation against the golden set."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from poc.evals.runner import EvalCase, EvalResult
    from poc.retrieval.hybrid import HybridRetriever

app = typer.Typer(help="Run baseline RAG evaluation.")
_log = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_DEFAULT_EVAL_SET = _PROJECT_ROOT / "packages" / "evals" / "data" / "golden_set_v2.jsonl"
_DEFAULT_DOCS_EVALS = _PROJECT_ROOT / "docs" / "evals"

_NON_LLM_METRICS = {"citation_precision", "groundedness_heuristic"}
_LLM_METRICS = {"faithfulness", "answer_relevance", "context_precision", "context_recall"}
_ALL_METRICS = _NON_LLM_METRICS | _LLM_METRICS


def latency_cost_summary(results: list[EvalResult]) -> dict[str, float | int]:
    """p50/p95 answer latency and per-answer cost over the successful cases."""
    ok = [r for r in results if not r.error]
    latencies = sorted(r.latency_ms for r in ok)
    costs = [r.cost_usd for r in ok if r.cost_usd is not None]

    def pct(q: float) -> int:
        if not latencies:
            return 0
        # nearest-rank percentile
        return latencies[max(0, math.ceil(q * len(latencies)) - 1)]

    return {
        "latency_p50_ms": pct(0.50),
        "latency_p95_ms": pct(0.95),
        "mean_cost_usd": round(sum(costs) / len(costs), 6) if costs else 0.0,
        "total_cost_usd": round(sum(costs), 6),
    }


@app.command()
def main(
    eval_set: Path = typer.Option(  # noqa: B008
        _DEFAULT_EVAL_SET,
        "--golden-set",
        help="Path to golden set JSONL",
    ),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant URL"),
    tenant: str = typer.Option("default", help="Tenant name"),
    provider: str | None = typer.Option(
        None, help="LLM provider name (default: POC_DEFAULT_PROVIDER, falls back to gemini)"
    ),
    model: str | None = typer.Option(
        None, help="Model name (default: the resolved provider's default model)"
    ),
    output: Path | None = typer.Option(None, "--output", help="Output JSON report path"),  # noqa: B008
    baseline: Path | None = typer.Option(  # noqa: B008
        None,
        "--baseline",
        help="Previous run JSON to compare against",
    ),
    data_path: Path = typer.Option(  # noqa: B008
        Path("data/raw/reviews.jsonl"),
        "--data-path",
        help="Path to the corpus JSONL used to build the BM25 index",
    ),
    limit: int | None = typer.Option(None, help="Max eval cases to run"),
    bm25_limit: int | None = typer.Option(
        None,
        "--bm25-limit",
        help="Index only the first N corpus documents into BM25 (default: whole corpus)",
    ),
    judge: bool = typer.Option(True, help="Run faithfulness LLM judge (legacy flag)"),
    use_agent: bool = typer.Option(
        False,
        "--agent/--no-agent",
        help=(
            "Use full LangGraph agent (intent→plan→retrieve→cluster→summarize→judge) "
            "instead of RAGPipeline baseline"
        ),
    ),
    metrics: str = typer.Option(
        "citation_precision,groundedness_heuristic",
        "--metrics",
        help=(
            "Comma-separated metrics to compute. "
            "Non-LLM: citation_precision,groundedness_heuristic. "
            "LLM-based: faithfulness,answer_relevance,context_precision,context_recall."
        ),
    ),
    retrieval_only: bool = typer.Option(
        False,
        "--retrieval-only",
        help=(
            "Skip the LLM provider and answer generation entirely; only run retrieval "
            "and compute a hit-rate gate. Works fully offline, no API keys required."
        ),
    ),
    min_hit_rate: float = typer.Option(
        0.8,
        "--min-hit-rate",
        help="Minimum acceptable hit_rate in --retrieval-only mode; exits 1 if not met",
    ),
) -> None:
    logging.basicConfig(level=logging.WARNING)

    # Resolve short name to full path (e.g. "golden_set_v2" → .../data/golden_set_v2.jsonl)
    if not eval_set.exists() and not eval_set.suffix:
        eval_set = _PROJECT_ROOT / "packages" / "evals" / "data" / f"{eval_set.name}.jsonl"

    # Resolve output path
    resolved_output = output
    if resolved_output is None:
        _DEFAULT_DOCS_EVALS.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(UTC).strftime("%Y%m%d")
        set_name = eval_set.stem  # e.g. "golden_set_v2"
        resolved_output = _DEFAULT_DOCS_EVALS / f"{date_str}_{set_name}.json"

    # Resolve provider/model: single source of default is LLMSettings.default_provider
    # (env POC_DEFAULT_PROVIDER, fallback "gemini") + that provider's registered model.
    from poc.llm.registry import default_model_for
    from poc.llm.settings import LLMSettings

    resolved_provider = provider or LLMSettings().default_provider
    resolved_model = model or default_model_for(resolved_provider)

    resolved_data_path = data_path if data_path.is_absolute() else _PROJECT_ROOT / data_path

    asyncio.run(
        _run_eval(
            eval_set=eval_set,
            qdrant_url=qdrant_url,
            tenant=tenant,
            provider=resolved_provider,
            model=resolved_model,
            output=resolved_output,
            baseline=baseline,
            data_path=resolved_data_path,
            limit=limit,
            bm25_limit=bm25_limit,
            run_judge=judge,
            use_agent=use_agent,
            metric_names=[m.strip() for m in metrics.split(",") if m.strip()],
            retrieval_only=retrieval_only,
            min_hit_rate=min_hit_rate,
        )
    )


async def _run_eval(
    *,
    eval_set: Path,
    qdrant_url: str,
    tenant: str,
    provider: str,
    model: str,
    output: Path,
    baseline: Path | None,
    data_path: Path,
    limit: int | None,
    bm25_limit: int | None,
    run_judge: bool,
    use_agent: bool,
    metric_names: list[str],
    retrieval_only: bool = False,
    min_hit_rate: float = 0.8,
) -> None:
    from poc.core.models import Question
    from poc.evals.metrics import (
        answer_relevance,
        citation_precision,
        faithfulness,
        groundedness_heuristic,
    )
    from poc.evals.metrics import (
        context_precision as ctx_precision,
    )
    from poc.evals.metrics import (
        context_recall as ctx_recall,
    )
    from poc.evals.runner import EvalResult, _check_substrings, load_eval_cases
    from poc.llm.openai_compatible import OpenAICompatibleProvider
    from poc.observability import langfuse_client
    from poc.observability.tracing import configure as configure_tracing
    from poc.observability.tracing import get_tracer
    from poc.retrieval.factory import build_hybrid_retriever

    # Validate metric names
    unknown = set(metric_names) - _ALL_METRICS
    if unknown:
        typer.echo(f"WARNING: unknown metrics (will skip): {unknown}", err=True)
        metric_names = [m for m in metric_names if m in _ALL_METRICS]

    active_llm_metrics = set(metric_names) & _LLM_METRICS

    # Initialise OpenTelemetry -> Langfuse export. The API app calls this at startup,
    # but this CLI does not go through it, so without this the get_tracer()/span calls
    # in the LLM and agent layers use the global no-op provider and export nothing.
    configure_tracing(service_name="run-eval")

    llm = None
    if not retrieval_only:
        typer.echo("Loading LLM provider...")
        llm = OpenAICompatibleProvider.from_env(provider)

    typer.echo("Loading eval cases...")
    cases = load_eval_cases(eval_set)
    if limit:
        cases = cases[:limit]
    if retrieval_only:
        typer.echo(f"Running {len(cases)} eval cases in retrieval-only mode")
    else:
        typer.echo(f"Running {len(cases)} eval cases with metrics: {metric_names}")

    typer.echo(f"Building retriever over {data_path}...")
    retriever, retriever_info = build_hybrid_retriever(
        data_path, qdrant_url=qdrant_url, tenant=tenant, bm25_limit=bm25_limit
    )
    typer.echo(
        f"Retriever: {retriever_info.mode} (bm25={retriever_info.bm25_chunks} chunks, "
        f"dense={retriever_info.dense_points} points, reranker={retriever_info.reranker})"
    )
    has_context = retriever_info.bm25_chunks > 0

    if retrieval_only:
        await _run_retrieval_only(
            cases=cases,
            retriever=retriever,
            eval_set=eval_set,
            output=output,
            min_hit_rate=min_hit_rate,
        )
        return

    if use_agent:
        from poc.agent.agent import ProductInsightAgent

        pipeline = ProductInsightAgent(llm=llm, retriever=retriever)
        typer.echo("Pipeline: ProductInsightAgent (LangGraph)")
    else:
        from poc.agent.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(retriever=retriever, llm=llm, model=model, top_k=8)
        typer.echo("Pipeline: RAGPipeline (baseline)")

    tracer = get_tracer()
    eval_run_id = f"eval-{eval_set.stem}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"

    results: list[EvalResult] = []
    # Per-metric accumulators: metric_name -> list[float]
    metric_accumulator: dict[str, list[float]] = {m: [] for m in metric_names}
    substring_matches: list[bool] = []

    for i, case in enumerate(cases):
        typer.echo(f"[{i + 1}/{len(cases)}] {case.id}: {case.question[:60]}...")
        # One Langfuse trace per case: the pipeline's node/LLM spans nest under
        # `eval.case`, all cases of this run share a session, and every metric is
        # attached as a score — so a run can be compared case by case in the UI.
        with tracer.start_as_current_span("eval.case") as case_span:
            langfuse_client.set_trace_attributes(
                case_span,
                name=f"eval.{eval_set.stem}.{case.id}",
                session_id=eval_run_id,
                tags=["eval", eval_set.stem, "agent" if use_agent else "baseline"],
                input={"case_id": case.id, "question": case.question},
            )
            try:
                t0 = time.monotonic()
                if use_agent:
                    answer = await pipeline.get_answer(
                        case.question, tenant=tenant, filters=case.filters or None
                    )
                else:
                    q = Question(text=case.question, filters=case.filters)
                    answer = await pipeline.answer(q)
                # Wall-clock answer latency (retrieval + all LLM calls), excluding metrics.
                answer_latency_ms = int((time.monotonic() - t0) * 1000)
                cited_ids = [c.chunk_id for c in answer.citations]

                context_texts: list[str] = []
                if has_context and active_llm_metrics:
                    retrieved = retriever.retrieve(case.question, top_k=5)
                    context_texts = [sc.chunk.text for sc in retrieved]

                # Compute requested metrics
                case_scores: dict[str, float] = {}

                if "citation_precision" in metric_names:
                    cp = citation_precision(cited_ids, case.expected_chunk_ids)
                    case_scores["citation_precision"] = cp.score

                if "groundedness_heuristic" in metric_names:
                    gh = groundedness_heuristic(answer.text, context_texts)
                    case_scores["groundedness_heuristic"] = gh.score

                if "faithfulness" in metric_names and (run_judge or "faithfulness" in metric_names):
                    if context_texts:
                        faith = await faithfulness(answer.text, context_texts, llm, model=model)
                        case_scores["faithfulness"] = faith.score
                    else:
                        case_scores["faithfulness"] = 0.0

                if "answer_relevance" in metric_names:
                    ar = await answer_relevance(case.question, answer.text, llm, model=model)
                    case_scores["answer_relevance"] = ar.score

                if "context_precision" in metric_names:
                    if context_texts:
                        cp_llm = await ctx_precision(case.question, context_texts, llm, model=model)
                        case_scores["context_precision"] = cp_llm.score
                    else:
                        case_scores["context_precision"] = 1.0

                if "context_recall" in metric_names:
                    expected_answer = " ".join(case.expected_answer_substrings)
                    if context_texts and expected_answer:
                        cr = await ctx_recall(
                            answer.text, expected_answer, context_texts, llm, model=model
                        )
                        case_scores["context_recall"] = cr.score
                    else:
                        case_scores["context_recall"] = 1.0 if not expected_answer else 0.0

                substr_match = _check_substrings(answer.text, case.expected_answer_substrings)

                # Legacy EvalResult for backward compat
                result = EvalResult(
                    case_id=case.id,
                    question=case.question,
                    answer_text=answer.text,
                    cited_chunk_ids=cited_ids,
                    faithfulness_score=case_scores.get("faithfulness", 0.0),
                    citation_precision_score=case_scores.get("citation_precision", 0.0),
                    substring_match=substr_match,
                    latency_ms=answer_latency_ms,
                    cost_usd=answer.cost_usd,
                )
                # Attach extended scores for output
                result._scores = case_scores  # type: ignore[attr-defined]
                result._retrieved_count = len(context_texts)  # type: ignore[attr-defined]

            except Exception as exc:
                _log.exception("Error on case %s", case.id)
                result = EvalResult(
                    case_id=case.id,
                    question=case.question,
                    answer_text="",
                    cited_chunk_ids=[],
                    faithfulness_score=0.0,
                    citation_precision_score=0.0,
                    substring_match=False,
                    latency_ms=0,
                    error=str(exc),
                )
                result._scores = {}  # type: ignore[attr-defined]
                result._retrieved_count = 0  # type: ignore[attr-defined]

            langfuse_client.set_trace_attributes(case_span, output=result.answer_text)
            for m_name, m_value in getattr(result, "_scores", {}).items():
                langfuse_client.score_current_trace(m_name, m_value)
            langfuse_client.score_current_trace(
                "substring_match", 1.0 if result.substring_match else 0.0
            )

        results.append(result)
        substring_matches.append(result.substring_match)
        for m_name in metric_names:
            score_val = getattr(result, "_scores", {}).get(m_name)
            if score_val is not None:
                metric_accumulator[m_name].append(score_val)

    # Summary stats
    n = len(results)
    errors = sum(1 for r in results if r.error)
    substr_rate = sum(1 for m in substring_matches if m) / n if n else 0.0

    metrics_summary: dict[str, dict[str, float]] = {}
    for m_name, scores in metric_accumulator.items():
        if scores:
            metrics_summary[m_name] = {
                "mean": round(sum(scores) / len(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
            }

    per_case = [
        {
            "id": r.case_id,
            "question": r.question,
            "scores": getattr(r, "_scores", {}),
            "retrieved_count": getattr(r, "_retrieved_count", 0),
            "substring_match": r.substring_match,
            "latency_ms": r.latency_ms,
            "cost_usd": r.cost_usd,
            "error": r.error,
        }
        for r in results
    ]

    timestamp = datetime.now(UTC).isoformat()
    golden_set_name = eval_set.stem

    report = {
        "timestamp": timestamp,
        "golden_set": golden_set_name,
        "pipeline": "agent" if use_agent else "baseline",
        "metrics_summary": metrics_summary,
        "summary": {
            "total_cases": n,
            "errors": errors,
            "substring_match_rate": round(substr_rate, 4),
            **latency_cost_summary(results),
        },
        "provider": provider,
        "model": model,
        "per_case": per_case,
    }

    langfuse_client.flush()

    typer.echo("\n=== Eval Report ===")
    typer.echo(f"Total cases:         {n}")
    typer.echo(f"Errors:              {errors}")
    typer.echo(f"Substring match:     {substr_rate:.4f}")
    lc = report["summary"]
    typer.echo(f"Latency p50/p95 ms:  {lc['latency_p50_ms']} / {lc['latency_p95_ms']}")
    typer.echo(
        f"Cost per answer:     ${lc['mean_cost_usd']:.5f} (total ${lc['total_cost_usd']:.4f})"
    )
    for m_name, stats in metrics_summary.items():
        typer.echo(
            f"{m_name}: mean={stats['mean']:.4f} min={stats['min']:.4f} max={stats['max']:.4f}"
        )

    # Baseline comparison
    if baseline and baseline.exists():
        typer.echo("\n=== Baseline Comparison ===")
        try:
            prev = json.loads(baseline.read_text(encoding="utf-8"))
            prev_summary = prev.get("metrics_summary", {})
            for m_name, stats in metrics_summary.items():
                if m_name in prev_summary:
                    delta = stats["mean"] - prev_summary[m_name]["mean"]
                    sign = "+" if delta >= 0 else ""
                    typer.echo(f"{m_name}: {sign}{delta:+.4f} vs baseline")
            report["baseline"] = str(baseline)
            report["baseline_deltas"] = {
                m: round(metrics_summary[m]["mean"] - prev_summary[m]["mean"], 4)
                for m in metrics_summary
                if m in prev_summary
            }
        except Exception as exc:
            typer.echo(f"Could not load baseline: {exc}", err=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"\nReport saved to: {output}")

    # Flush buffered spans before the process exits. BatchSpanProcessor sends in
    # batches; a short-lived CLI can terminate before the last batch is shipped.
    from opentelemetry import trace

    trace.get_tracer_provider().force_flush()


def compute_hit(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> bool:
    """True if at least one expected chunk id was retrieved."""
    if not expected_chunk_ids:
        return False
    return bool(set(retrieved_chunk_ids) & set(expected_chunk_ids))


async def _run_retrieval_only(
    *,
    cases: list[EvalCase],
    retriever: HybridRetriever,
    eval_set: Path,
    output: Path,
    min_hit_rate: float,
    top_k: int = 10,
) -> None:
    """Retrieval-only eval mode: no LLM, no answer generation, no network required.

    For each case, run hybrid retrieval (BM25-only when Qdrant is unavailable) and
    check whether at least one of the case's `expected_chunk_ids` was retrieved.
    Reports the overall hit_rate and exits non-zero if it falls below `min_hit_rate`
    — this is the keyless CI gate.
    """
    per_case: list[dict] = []
    hits = 0

    for i, case in enumerate(cases):
        typer.echo(f"[{i + 1}/{len(cases)}] {case.id}: {case.question[:60]}...")
        scored = retriever.retrieve(case.question, top_k=top_k, filters=case.filters or None)
        retrieved_ids = [sc.chunk.id for sc in scored]
        hit = compute_hit(retrieved_ids, case.expected_chunk_ids)
        if hit:
            hits += 1
        per_case.append(
            {
                "id": case.id,
                "question": case.question,
                "hit": hit,
                "retrieved_count": len(retrieved_ids),
                "retrieved_chunk_ids": retrieved_ids,
            }
        )

    n = len(cases)
    hit_rate = round(hits / n, 4) if n else 0.0

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "golden_set": eval_set.stem,
        "mode": "retrieval_only",
        "summary": {
            "total_cases": n,
            "hits": hits,
            "hit_rate": hit_rate,
            "min_hit_rate": min_hit_rate,
        },
        "per_case": per_case,
    }

    typer.echo("\n=== Retrieval-only Eval Report ===")
    typer.echo(f"Total cases: {n}")
    typer.echo(f"Hits:        {hits}")
    typer.echo(f"Hit rate:    {hit_rate:.4f} (min required: {min_hit_rate:.4f})")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"\nReport saved to: {output}")

    if hit_rate < min_hit_rate:
        typer.echo(
            f"\nFAIL: hit_rate {hit_rate:.4f} is below the required minimum {min_hit_rate:.4f}",
            err=True,
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
