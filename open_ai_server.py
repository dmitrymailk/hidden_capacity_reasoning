import asyncio
import json
import time

from typing import Optional, List

from pydantic import BaseModel, Field

from starlette.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Request
import uuid

app = FastAPI(title="OpenAI-compatible API")


# data models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.messages:
        resp_content = (
            "As a mock AI Assitant, I can only echo your last message: "
            + request.messages[-1].content
        )
    else:
        resp_content = "As a mock AI Assitant, I can only echo your last message, but there wasn't one!"

    return {
        "id": uuid.uuid4(),
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [
            {
                "message": Message(role="assistant", content=resp_content),
                "index": 0,
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
