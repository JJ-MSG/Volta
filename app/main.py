from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.chat import generate_chat_stream
from app.models import ChatRequest

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        generate_chat_stream(request),
        media_type="text/event-stream",
    )