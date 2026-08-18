import asyncio

from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.chat import generate_chat_stream
from app.models import ChatRequest


client = TestClient(app)


def parse_sse_events(response):
    events = []
    current_event = None
    current_data = None

    for line in response.iter_lines():
        if not line:
            if current_event is not None:
                events.append(
                    {
                        "event": current_event,
                        "data": current_data,
                    }
                )
                current_event = None
                current_data = None

            continue

        if line.startswith("event:"):
            current_event = line.removeprefix("event:").strip()

        elif line.startswith("data:"):
            current_data = line.removeprefix("data:").strip()

    return events


def test_chat_stream_success():
    async def fake_stream(content, model):
        for chunk in ["R", "AG", " is", " a", " technique"]:
            yield chunk

    with patch("app.chat.bedrock_llm_stream", fake_stream):
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "chat_id": "123",
                "content": "Explain RAG",
                "model": "nova-pro"
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse_events(response)

    assert events[0]["event"] == "metadata"

    token_events = [
        event
        for event in events
        if event["event"] == "token"
    ]

    assert len(token_events) > 0
    assert events[-1]["event"] == "done"


def test_chat_stream_event_order():
    async def fake_stream(content, model):
        for chunk in ["R", "AG", " is", " a"]:
            yield chunk

    with patch("app.chat.bedrock_llm_stream", fake_stream):
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "chat_id": "123",
                "content": "Explain RAG",
                "model": "nova-pro"
            },
        )

    events = parse_sse_events(response)

    event_types = [
        event["event"]
        for event in events
    ]

    assert event_types[0] == "metadata"
    assert all(
        event_type == "token"
        for event_type in event_types[1:-1]
    )
    assert event_types[-1] == "done"


def test_chat_stream_missing_content():
    response = client.post(
        "/api/v1/chat/stream",
        json={
            "chat_id": "123",
            "model": "nova-pro",
        },
    )

    assert response.status_code == 422


def test_chat_stream_missing_chat_id():
    response = client.post(
        "/api/v1/chat/stream",
        json={
            "content": "Explain RAG",
            "model": "nova-pro",
        },
    )

    assert response.status_code == 422


def test_chat_stream_mid_stream_failure():
    async def failing_stream(content, model):
        yield "R"
        yield "AG"
        yield " is"
        raise RuntimeError("Simulated LLM failure")

    with patch("app.chat.bedrock_llm_stream", failing_stream):
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "chat_id": "123",
                "content": "Explain RAG",
                "model": "nova-pro"
            },
        )

    events = parse_sse_events(response)

    event_types = [
        event["event"]
        for event in events
    ]

    assert event_types == [
        "metadata",
        "token",
        "token",
        "token",
        "error",
    ]


def test_chat_stream_cancellation():
    cancellation_detected = False

    async def cancellable_stream(content, model):
        nonlocal cancellation_detected

        try:
            while True:
                yield "test chunk"
                await asyncio.sleep(10)

        except asyncio.CancelledError:
            cancellation_detected = True
            raise

    async def run_test():
        nonlocal cancellation_detected

        request = ChatRequest(
            chat_id="123",
            content="Explain RAG",
            model="nova-pro"
        )

        with patch("app.chat.bedrock_llm_stream", cancellable_stream):
            stream = generate_chat_stream(request)

            async def consume():
                async for _ in stream:
                    pass

            task = asyncio.create_task(consume())

            # Give the consumer a chance to start.
            await asyncio.sleep(0)

            # Simulate cancellation caused by client disconnect.
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_test())

    assert cancellation_detected is True
    