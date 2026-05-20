from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, login
import json
import time
import os

# Login to HF Hub
token = os.getenv("HF_TOKEN")
if token:
    login(token)

app = FastAPI(title="Chat Doctor API", version="1.0.0")

# Download model on startup
print("Downloading model...")
model_path = hf_hub_download(
    repo_id="SatRag/chat-doctor-gguf",
    filename="chat-doctor.gguf",
    repo_type="model",
)
print(f"Model loaded: {model_path}")

# Initialize llama.cpp
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=4,
    verbose=False,
    # Gemma uses a specific chat format - llama-cpp-python handles this
    chat_format="gemma",  # or try "gemma-2" if this doesn't work
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 300
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@app.get("/health")
async def health():
    return {"status": "ok", "model": "chat-doctor-gguf"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    messages = [m.dict() for m in request.messages]
    
    if request.stream:
        def event_stream():
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            )
            for chunk in output:
                delta = chunk["choices"][0]["delta"]
                if delta.get("content"):
                    data = {
                        "choices": [{
                            "delta": {"content": delta["content"]},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "chat-doctor-gguf",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output["choices"][0]["message"]["content"].strip()
            },
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)