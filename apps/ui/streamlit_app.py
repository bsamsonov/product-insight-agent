from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("POC_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Product Insight Agent", layout="wide")
st.title("Product Insight Agent — Demo")

# Sidebar
with st.sidebar:
    api_url = st.text_input("API URL", value=API_URL)
    tenant = st.text_input("Tenant", value="default")
    model = st.text_input("Model (optional)", value="")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            if meta.get("needs_human_review"):
                st.warning("Low confidence: flagged for human review (HITL).")
            with st.expander("Details"):
                st.caption(f"Model: {meta.get('model') or '—'}")
                cols = st.columns(4)
                cols[0].metric("Latency", f"{meta.get('latency_ms', 0)} ms")
                cost = meta.get("cost_usd")
                cols[1].metric("Cost", f"${cost:.4f}" if cost else "—")
                judge = meta.get("judge_score")
                cols[2].metric("Judge", f"{judge:.2f}" if judge is not None else "—")
                grounded = meta.get("groundedness")
                cols[3].metric("Groundedness", f"{grounded:.2f}" if grounded is not None else "—")
                if meta.get("citations"):
                    st.markdown("**Citations**")
                    for c in meta["citations"]:
                        excerpt = " ".join(str(c.get("text_excerpt", "")).split())[:160]
                        st.markdown(f"- `{c.get('chunk_id')}` — {excerpt}")

# Input
question = st.chat_input("Ask about product reviews...")
if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    # Call API
    payload: dict[str, object] = {"question": question}
    if model:
        payload["model"] = model

    with st.spinner("Thinking..."):
        try:
            resp = httpx.post(
                f"{api_url}/ask",
                json=payload,
                headers={"X-Tenant-Id": tenant},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()

            # Guardrail refusals come back as 200 {status: refused, reason: ...}
            # (S4.T5) with no `answer` field — surface the reason instead of a
            # misleading "No answer returned."
            if data.get("status") == "refused":
                answer = f"⚠️ Request refused by guardrails ({data.get('reason', 'unknown')})."
            else:
                answer = data.get("answer", "No answer returned.")
            metadata = {
                "model": data.get("model", ""),
                "latency_ms": data.get("latency_ms", 0),
                "cost_usd": data.get("cost_usd"),
                "citations": data.get("citations", []),
                "judge_score": data.get("judge_score"),
                "groundedness": data.get("groundedness"),
                "needs_human_review": data.get("needs_human_review", False),
            }

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata,
                }
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.json() if e.response.content else {}
            st.error(f"API error {e.response.status_code}: {error_body.get('detail', str(e))}")
        except httpx.RequestError as e:
            st.error(f"Connection error: {e}. Is the API running at {api_url}?")

    st.rerun()
