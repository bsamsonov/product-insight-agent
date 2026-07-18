from __future__ import annotations

import asyncio
import sys

import typer
from poc.llm.openai_compatible import OpenAICompatibleProvider
from poc.llm.provider import LLMMessage
from poc.llm.registry import default_model_for
from poc.llm.settings import LLMSettings

app = typer.Typer(add_completion=False)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the LLM"),
    provider: str | None = typer.Option(
        None,
        help="LLM provider: gemini|groq|deepseek|ollama|omniroute (default: POC_DEFAULT_PROVIDER)",
    ),
    model: str | None = typer.Option(
        None, help="Model name (default: the resolved provider's default model)"
    ),
    max_tokens: int = typer.Option(1024, help="Maximum tokens in response"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream response token by token"),
) -> None:
    """Ask a question using the configured LLM provider."""
    settings = LLMSettings()
    resolved_provider = provider or settings.default_provider
    resolved_model = model or default_model_for(resolved_provider)

    llm = OpenAICompatibleProvider.from_env(resolved_provider)
    messages = [LLMMessage(role="user", content=question)]

    if stream:

        async def _stream() -> None:
            gen = await llm.stream(
                messages, model=resolved_model, max_tokens=max_tokens, temperature=0.7
            )
            async for chunk in gen:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")

        asyncio.run(_stream())
    else:
        response = asyncio.run(
            llm.complete(messages, model=resolved_model, max_tokens=max_tokens, temperature=0.7)
        )
        typer.echo(response.content)


if __name__ == "__main__":
    app()
