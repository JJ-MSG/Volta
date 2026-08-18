from pydantic import BaseModel, field_validator

from app.config import MODEL_REGISTRY


class ChatRequest(BaseModel):
    chat_id: str
    content: str
    model: str

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value not in MODEL_REGISTRY:
            raise ValueError("Unsupported model")

        return value