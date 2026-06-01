from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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
    MODELS,
    DEFAULT_MODEL,
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
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chat-doc-bot.vercel.app",  # production frontend
        "http://localhost:5173",             # Local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load ALL Models on Startup
# =========================================================

loaded_models = {}

def load_model(model_id):
    """Load a model if not already loaded."""
    if model_id in loaded_models:
        return loaded_models[model_id]
    
    cfg = MODELS.get(model_id)
    if not cfg:
        raise ValueError(f"Unknown model: {model_id}")
    
    print(f"Loading {model_id} from {cfg['repo']}...")
    
    model_path = hf_hub_download(
        repo_id=cfg["repo"],
        filename=cfg["file"],
    )
    
    print(f"  Downloaded: {model_path}")
    print(f"  Size: {os.path.getsize(model_path) / 1e9:.2f} GB")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,
    )
    
    loaded_models[model_id] = llm
    print(f"  Loaded successfully!")
    return llm

# Load default model on startup
print("=" * 50)
print("Loading default model...")
load_model(DEFAULT_MODEL)
print("=" * 50)

# =========================================================
# Request Models
# =========================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = DEFAULT_MODEL  # User selects "chat-doctor-q4" or "chat-doctor-q8"
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    temperature: Optional[float] = TEMPERATURE
    stream: Optional[bool] = False


# =========================================================
# Helpers
# =========================================================

def clean_response(text: str) -> str:
    # Remove special tokens
    text = re.sub(r"<\|.*?\|>", "", text)
    
    # Fix numbered lists: "1.text" -> "1. text"
    text = re.sub(r"(\d+)\.([A-Za-z])", r"\1. \2", text)
    
    # Ensure newlines before list items
    text = re.sub(r"(\d+\.)", r"\n\1", text)
    
    # Fix space after periods (but not decimal points)
    text = re.sub(r"\.([A-Z])", r". \1", text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
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
@app.head("/health")  # For Uptime Robot
async def health():
    return {
        "status": "ok",
        "models_loaded": list(loaded_models.keys()),
        "default_model": DEFAULT_MODEL,
    }


@app.get("/v1/models")
async def list_models():
    """List available models with metadata."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1779244265,
                "owned_by": "SatRag",
                "name": cfg["name"],
                "description": cfg["description"],
                "quantization": cfg["quantization"],
                "loaded": model_id in loaded_models,
            }
            for model_id, cfg in MODELS.items()
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, authorization: str = Header(None)):
    verify_key(authorization)
    
    # Validate model selection
    if request.model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {', '.join(MODELS.keys())}"
        )
    
    # Load model if not already loaded (lazy load)
    try:
        llm = load_model(request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
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
        "model": request.model,
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