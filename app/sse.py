import json


def sse_event(event: str, data: dict) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data)}\n\n"
    )


def metadata_event(chat_id: str) -> str:
    return sse_event(
        "metadata",
        {"chat_id": chat_id},
    )


def token_event(text: str) -> str:
    return sse_event(
        "token",
        {"text": text},
    )


def done_event(finish_reason: str = "stop") -> str:
    return sse_event(
        "done",
        {"finish_reason": finish_reason},
    )


def error_event(code: str, message: str) -> str:
    return sse_event(
        "error",
        {
            "code": code,
            "message": message,
        },
    )