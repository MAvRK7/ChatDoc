from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, login
from starlette.status import HTTP_401_UNAUTHORIZED

import json
import time
import os
import re

from config import (
    MODEL_REPO,
    MODEL_FILE,
    SYSTEM_PROMPT,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    REPETITION_PENALTY,
    TOP_P,
    N_CTX,
    N_THREADS,
    API_KEYS,
    STOP_TOKENS,
)

# =========================================================
# Hugging Face Login
# =========================================================

token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(
    title="Chat Doctor API",
    version="1.0.0"
)

# =========================================================
# Download GGUF Model
# =========================================================

print(f"Downloading model from {MODEL_REPO}/{MODEL_FILE}...")

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
)

print(f"Model downloaded at: {model_path}")

# =========================================================
# Load Llama Model
# =========================================================

# Replace the llm = Llama(...) block with this:

print(f"Loading model: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
print(f"File size: {os.path.getsize(model_path) / 1e9:.2f} GB")

try:
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=True,  # Enable verbose logging
    )
    print("Model loaded successfully!")
except Exception as e:
    import traceback
    print("=" * 50)
    print("MODEL LOAD FAILED - REAL ERROR:")
    print(traceback.format_exc())
    print("=" * 50)
    raise RuntimeError(f"Failed to load model: {e}") from e

#-------------
#DEBUG
#------------

print(f"CPU count: {os.cpu_count()}")
print(f"Using threads: {N_THREADS}")
print(f"Model: {MODEL_FILE}")
print(f"Model size: {os.path.getsize(model_path) / 1e9:.2f} GB")

# =========================================================
# Request Models
# =========================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    temperature: Optional[float] = TEMPERATURE
    stream: Optional[bool] = False


# =========================================================
# Helpers
# =========================================================

def clean_response(text: str) -> str:
    # Remove special tokens
    text = re.sub(r"<\|.*?\|>", "", text)

    # Add spacing after punctuation if missing
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

    # Proper numbered list formatting
    text = re.sub(r'(\d+)\.\s*', r'\n\n\1. ', text)

    # Add line breaks before bullet points
    text = re.sub(r'[-•]\s*', r'\n- ', text)

    # Add spacing after colons before lists
    text = re.sub(r':\s*(\d+\.)', r':\n\n\1', text)

    # Fix merged questions
    text = re.sub(r'\?([A-Z])', r'?\n\n\1', text)

    # Collapse excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def verify_key(authorization: str):
    if not authorization:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    key = authorization[7:] if authorization.startswith("Bearer ") else authorization
    if key not in API_KEYS:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid API key")


# =========================================================
# Endpoints
# =========================================================

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_FILE.replace(".gguf", "")}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_FILE.replace(".gguf", ""),
            "object": "model",
            "created": 1779244265,
            "owned_by": "SatRag",
            "description": "Medical chat assistant fine-tuned on Gemma 4 E2B",
            "base_model": MODEL_FILE.replace(".gguf", ""),
            "parameters": "4B",
            "quantization": "Q4_K_M" if "q4" in MODEL_FILE.lower() else "Q8_0",
            "context_length": N_CTX,
            "architecture": "Gemma 4",
        }]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, authorization: str = Header(None)):
    verify_key(authorization)
    
    messages = [m.dict() for m in request.messages]

    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, {
            "role": "system",
            "content": SYSTEM_PROMPT,
        })

    if request.stream:
        def event_stream():
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                repeat_penalty=REPETITION_PENALTY,
                top_p=TOP_P,
                stop=STOP_TOKENS,
                stream=True,
            )
            for chunk in output:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    content = re.sub(r"<\|.*?\|>", "", content)
                    if content.strip():
                        data = {
                            "choices": [{
                                "delta": {"content": content},
                                "finish_reason": None,
                            }]
                        }
                        yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        repeat_penalty=REPETITION_PENALTY,
        top_p=TOP_P,
        stop=STOP_TOKENS,
    )

    reply = clean_response(output["choices"][0]["message"]["content"])

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_FILE.replace(".gguf", ""),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": reply,
            },
            "finish_reason": "stop",
        }],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
'''
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, login

import json
import time
import os
import re

# =========================================================
# Hugging Face Login
# =========================================================

token = os.getenv("HF_TOKEN")

if token:
    login(token=token)

# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(
    title="Chat Doctor API",
    version="1.0.0"
)

# =========================================================
# Download GGUF Model
# =========================================================

print("Downloading model...")

model_path = hf_hub_download(
    repo_id="SatRag/chat-doctor-gguf",
    filename="chat-doctor.gguf",
)

print(f"Model downloaded at: {model_path}")

# =========================================================
# Load Llama Model
# =========================================================

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False,
)

# =========================================================
# Request Models
# =========================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 160
    temperature: Optional[float] = 0.3
    stream: Optional[bool] = False


# =========================================================
# Helpers
# =========================================================

def clean_response(text: str) -> str:
    """
    Cleans unwanted tokens and formatting issues
    """

    # Remove leaked template tokens
    text = re.sub(r"<\|.*?\|>", "", text)

    # Add newline before numbered lists
    text = re.sub(r"(\d+\.)", r"\n\1", text)

    # Add newline after colon before lists
    text = re.sub(r":\n?1\.", ":\n\n1.", text)

    # Remove extra blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================
# Health Endpoint
# =========================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "chat-doctor-gguf"
    }


# =========================================================
# Chat Completion Endpoint
# =========================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):

    messages = [m.dict() for m in request.messages]

    # Add default system prompt if missing
    if not any(m["role"] == "system" for m in messages):
        messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "You are a concise and helpful medical assistant. "
                    "Keep responses short and clear."
                ),
            },
        )

    # =====================================================
    # Streaming Response
    # =====================================================

    if request.stream:

        def event_stream():

            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                repeat_penalty=1.1,
                top_p=0.9,
                stop=[
                    "<|turn|>",
                    "<|channel|>",
                    "<end_of_turn>",
                    "</s>",
                ],
                stream=True,
            )

            for chunk in output:

                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")

                if content:

                    # Remove leaked template tokens
                    content = re.sub(r"<\|.*?\|>", "", content)

                    if content.strip():

                        data = {
                            "choices": [
                                {
                                    "delta": {
                                        "content": content
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }

                        yield f"data: {json.dumps(data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )

    # =====================================================
    # Non-streaming Response
    # =====================================================

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        repeat_penalty=1.1,
        top_p=0.9,
        stop=[
            "<|turn|>",
            "<|channel|>",
            "<end_of_turn>",
            "</s>",
        ],
    )

    reply = output["choices"][0]["message"]["content"]
    reply = clean_response(reply)

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "chat-doctor-gguf",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply,
                },
                "finish_reason": "stop",
            }
        ],
    }


# =========================================================
# Run Server
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
    )
'''