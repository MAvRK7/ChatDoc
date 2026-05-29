import os

# === MODEL SELECTION ===
# Set via environment variables on HF Spaces, or change defaults here
# Q8 (higher quality, slower):  MODEL_REPO=SatRag/chat-doctor-gguf  MODEL_FILE=chat-doctor.gguf
# Q4 (faster, slightly lower quality): MODEL_REPO=SatRag/chat-doctor-q4  MODEL_FILE=chat-doctor-q4.gguf

# config.py
MODEL_REPO = os.getenv("MODEL_REPO", "SatRag/chat-doctor-q4")  # Default to Q4
MODEL_FILE = os.getenv("MODEL_FILE", "chat-doctor-q4.gguf")   # Default to Q4

# === MODEL INFO ===
BASE_MODEL = "google/gemma-4-E2B-it"
ADAPTER_REPO = "satvikraghav/chat-doctor-gemma4-lora"

# === GENERATION SETTINGS ===
SYSTEM_PROMPT = "You are ChatDoc, a helpful medical assistant. Keep answers concise, accurate, and complete. If unsure, say you don't know."

# Auto-detect threads based on CPU count
N_THREADS = 8  # This sets the variable

MAX_NEW_TOKENS = 160
TEMPERATURE = 0.3
REPETITION_PENALTY = 1.1
TOP_P = 0.9

N_CTX = 2048
N_THREADS = 4

# === AUTH ===
API_KEYS = {"test-key-123", "demo-key-456"}

# === STOP TOKENS ===
STOP_TOKENS = ["<|turn|>", "<|channel|>", "<end_of_turn>", "</s>"]
