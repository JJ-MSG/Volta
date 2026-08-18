import asyncio
from collections.abc import AsyncGenerator

from app.llm import bedrock_llm_stream
from app.models import ChatRequest
from app.sse import (
    metadata_event,
    token_event,
    done_event,
    error_event,
)


async def generate_chat_stream(
    request: ChatRequest,
) -> AsyncGenerator[str, None]:
    yield metadata_event(request.chat_id)

    try:
        async for chunk in bedrock_llm_stream(request.content, request.model):
            yield token_event(chunk)

        yield done_event()

    except asyncio.CancelledError:
        print("CLIENT DISCONNECTED - GENERATION CANCELLED")
        raise

    except Exception as exc:
        print(f"LLM STREAM ERROR: {exc!r}")

        yield error_event(
            "LLM_STREAM_ERROR",
            "The response could not be completed.",
        )