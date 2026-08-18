import asyncio
from collections.abc import AsyncGenerator

import boto3

from app.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    AWS_REGION,
    MODEL_REGISTRY
)


def create_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )


async def bedrock_llm_stream(
    content: str,
    model: str
) -> AsyncGenerator[str, None]:
    client = create_bedrock_client()

    response = await asyncio.to_thread(
        client.converse_stream,
        modelId = MODEL_REGISTRY[model],
        messages=[
            {
                "role": "user",
                "content": [{"text": content}],
            }
        ],
    )

    stream = response["stream"]
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def read_stream():
        try:
            for event in stream:
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get("delta", {})
                    text = delta.get("text")

                    if text:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            ("text", text),
                        )

                elif "messageStop" in event:
                    stop_reason = event["messageStop"]["stopReason"]

                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        ("done", stop_reason),
                    )

                    return

        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                ("error", exc),
            )

    reader_task = asyncio.create_task(
        asyncio.to_thread(read_stream)
    )

    try:
        while True:
            event_type, value = await queue.get()

            if event_type == "text":
                yield value

            elif event_type == "done":
                return

            elif event_type == "error":
                raise value

    except asyncio.CancelledError:
        print("CANCELLING BEDROCK STREAM")

        stream.close()
        reader_task.cancel()

        raise

    finally:
        if not reader_task.done():
            reader_task.cancel()