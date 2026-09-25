from __future__ import annotations

import logging
import time
from pathlib import Path

from poc.agent.citations import cited_chunk_ids
from poc.core.models import Answer, Citation, Question
from poc.llm.provider import LLMMessage, LLMProvider
from poc.retrieval.hybrid import HybridRetriever
from poc.retrieval.qdrant_index import ScoredChunk

_log = logging.getLogger(__name__)

# Matches citation markers like [chunk_42__c3] in answers


def _format_context(chunks: list[ScoredChunk]) -> str:
    parts = []
    for sc in chunks:
        parts.append(f"[{sc.chunk.id}]\n{sc.chunk.text}")
    return "\n\n---\n\n".join(parts)


def _extract_citations(answer_text: str, chunks: list[ScoredChunk]) -> list[Citation]:
    chunk_map = {sc.chunk.id: sc for sc in chunks}
    citations: list[Citation] = []
    for cid in cited_chunk_ids(answer_text):
        if cid in chunk_map:
            sc = chunk_map[cid]
            citations.append(
                Citation(
                    chunk_id=cid,
                    text_excerpt=sc.chunk.text[:200],
                    score=sc.score,
                )
            )
    return citations


class RAGPipeline:
    """Minimal RAG pipeline: question → retrieve → LLM answer with citations."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        llm: LLMProvider,
        model: str,
        max_tokens: int = 1024,
        top_k: int = 10,
        prompts_data_dir: Path | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._model = model
        self._max_tokens = max_tokens
        self._top_k = top_k
        self._prompts_data_dir = prompts_data_dir

    def _get_system_prompt(self) -> str:
        try:
            from poc.prompts.registry import PromptRegistry

            registry = PromptRegistry(self._prompts_data_dir)
            tmpl = registry.get("rag_baseline", "rag_baseline")
            return tmpl.system
        except Exception:
            _log.warning("Could not load prompt from registry, using inline fallback")
            return (
                "Answer using ONLY the provided context fragments. "
                "Cite each claim with [chunk_id]. If unsure, say 'UNKNOWN'."
            )

    def _get_user_prompt(self, question: str, context: str) -> str:
        try:
            from poc.prompts.registry import PromptRegistry

            registry = PromptRegistry(self._prompts_data_dir)
            tmpl = registry.get("rag_baseline", "rag_baseline")
            return tmpl.render_user(context=context, question=question)
        except Exception:
            return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    async def answer(
        self,
        question: Question,
    ) -> Answer:
        t0 = time.monotonic()

        # Retrieve
        chunks = self._retriever.retrieve(
            question.text,
            top_k=self._top_k,
            filters=question.filters or None,
        )

        if not chunks:
            return Answer(
                question=question.text,
                text="UNKNOWN: no relevant documents found.",
                model=self._model,
                latency_ms=int((time.monotonic() - t0) * 1000),
            )

        context = _format_context(chunks)
        messages = [
            LLMMessage(role="system", content=self._get_system_prompt()),
            LLMMessage(role="user", content=self._get_user_prompt(question.text, context)),
        ]

        response = await self._llm.complete(
            messages,
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        citations = _extract_citations(response.content, chunks)
        latency_ms = int((time.monotonic() - t0) * 1000)

        return Answer(
            question=question.text,
            text=response.content,
            citations=citations,
            used_chunks=[sc.chunk.id for sc in chunks],
            model=response.model,
            cost_usd=response.cost_usd,
            latency_ms=latency_ms,
        )
