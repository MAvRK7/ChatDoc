import os

BASE_MODEL = "google/gemma-4-E2B-it"
ADAPTER_REPO = "satvikraghav/chat-doctor-gemma4-lora"
# ADAPTER_REPO = os.getenv("ADAPTER_PATH", "checkpoints/archive")  # local 

SYSTEM_PROMPT = """You are ChatDoc, a helpful medical assistant. Keep answers concise, accurate, and complete. 
If unsure, say you don't know."""

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7
REPETITION_PENALTY = 1.15

# Simple API keys (replace with env vars in production)
API_KEYS = {"test-key-123", "demo-key-456"}
