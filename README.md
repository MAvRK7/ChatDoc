# ChatDoc — AI Medical Assistant

A production-ready medical chatbot API and React frontend, built on a fine-tuned Gemma 4 E2B model with dual quantization support for flexible deployment.

## What It Does

ChatDoc answers general medical questions with concise, accurate responses. It supports both fast (Q4) and accurate (Q8) inference modes, selectable at query time.

---
## Architecture

```text
┌─────────────┐      HTTP/REST      ┌─────────────────┐
│  React UI   │ ◄─────────────────► │  FastAPI (HF    │
│  (Vercel)   │   OpenAI-compatible │  Spaces Docker) │
└─────────────┘      streaming      └─────────────────┘
                                            │
                                     ┌──────┴──────┐
                                     │  llama.cpp  │
                                     │  Q4 / Q8    │
                                     │  GGUF       │
                                     └─────────────┘
```
---
## Tech Stack

```
| Layer          | Tech                                   |
| -------------- | -------------------------------------- |
| **Frontend**   | React 18, Vite, TailwindCSS            |
| **Backend**    | FastAPI, llama-cpp-python              |
| **Model**      | Gemma 4 E2B + LoRA fine-tune           |
| **Inference**  | llama.cpp (CPU-optimized)              |
| **Deployment** | Hugging Face Spaces (API), Vercel (UI) |
```
---
## Features

- Dual Model Support — Switch between Q4 (fast) and Q8 (accurate) at runtime
- Streaming Responses — Real-time token streaming with Server-Sent Events
- OpenAI-Compatible API — Drop-in replacement for /v1/chat/completions
- API Key Auth — Simple Bearer token authentication
- Premium UI — Glass morphism, animated gradients, responsive design
- Medical Disclaimer — Ethical AI usage notice on every interaction

## Dual Support:
```
| Variable     | Value                     | Effect                  |
| ------------ | ------------------------- | ----------------------- |
| `MODEL_REPO` | `SatRag/chat-doctor-q4`   | Uses Q4 model           |
| `MODEL_FILE` | `chat-doctor-q4.gguf`     | Q4 filename             |
| `MODEL_REPO` | `SatRag/chat-doctor-gguf` | Uses Q8 model (default) |
| `MODEL_FILE` | `chat-doctor.gguf`        | Q8 filename             |
```
---

## Quick Start

### Backend (HF Spaces)

1. Fork or clone this repo
2. Set environment variables in Space settings:
    - HF_TOKEN — your Hugging Face token
    - API_KEY — set your own key (default: test-key-123)
    - DEFAULT_MODEL — chat-doctor-q4 or chat-doctor-q8

3. Deploy to Hugging Face Spaces (Docker SDK)

### Frontend (Local)

```
cd chat-doctor-frontend
npm install
npm run dev
```

### Frontend (Production)
```
npm run build
vercel --prod
```
---
## API Usage

```
curl -X POST https://SatRag-chat-doctor-api.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key-123" \
  -d '{
    "model": "chat-doctor-q4",
    "messages": [{"role": "user", "content": "What causes migraines?"}],
    "stream": true,
    "max_tokens": 160,
    "temperature": 0.3
  }'

```
---
## Model Details
```
| Spec               | Value                                           |
| ------------------ | ----------------------------------------------- |
| Base Model         | `google/gemma-4-E2B-it`                         |
| Fine-tuning        | LoRA on medical Q\&A + curated instruction data |
| Parameters         | 4B                                              |
| Quantizations      | Q4\_K\_M (~2.5GB), Q8\_0 (~5GB)                 |
| Context Length     | 2048 tokens                                     |
| Training Framework | TRL SFTTrainer with assistant-only masking      |
```

## Project Structure 

```
├── app.py                    # FastAPI server
├── config.py                 # Model & generation config
├── Dockerfile                # HF Spaces build
├── requirements.txt
└── chat-doctor-frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── ChatMessage.jsx
    │   │   ├── ChatInput.jsx
    │   │   ├── Header.jsx
    │   │   ├── InfoModal.jsx
    │   │   ├── SuggestionCards.jsx
    │   │   └── ThinkingIndicator.jsx
    │   └── hooks/
    │       └── useChatStream.js
    ├── index.html
    ├── package.json
    └── tailwind.config.js
```
---
## Env Variables

```
| Variable        | Default          | Description                        |
| --------------- | ---------------- | ---------------------------------- |
| `HF_TOKEN`      | —                | Hugging Face auth token            |
| `API_KEY`       | `test-key-123`   | API authentication key             |
| `DEFAULT_MODEL` | `chat-doctor-q4` | Default model on startup           |
| `MODEL_REPO`    | —                | Override HF repo for custom models |
| `MODEL_FILE`    | —                | Override GGUF filename             |

```
---

## Limitations

- CPU-only inference — Responses take 5-15 seconds depending on length
- Not a substitute for professional medical advice — Always consult a doctor for serious symptoms
- General health Q&A only — No diagnosis, prescriptions, or emergency guidance

---

## Acknowledgments

- llama.cpp for efficient CPU inference
- Hugging Face for model hosting and Spaces
- Google for the Gemma 4 model family