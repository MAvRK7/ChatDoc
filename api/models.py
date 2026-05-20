from pydantic import BaseModel
from typing import List, Optional, Literal
import uuid  # add this import

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "chat-doctor-v1"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    top_p: Optional[float] = 0.9

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict