# ChatDoc — AI Medical Assistant

A production-ready medical chatbot API and React frontend, built on a fine-tuned Gemma 4 E2B model with dual quantization support for flexible deployment.

## What It Does

ChatDoc answers general medical questions with concise, accurate responses. It supports both fast (Q4) and accurate (Q8) inference modes, selectable at query time.

---
## Architecture

```
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


| Layer          | Tech                                   |
| -------------- | -------------------------------------- |
| **Frontend**   | React 18, Vite, TailwindCSS            |
| **Backend**    | FastAPI, llama-cpp-python              |
| **Model**      | Gemma 4 E2B + LoRA fine-tune           |
| **Inference**  | llama.cpp (CPU-optimized)              |
| **Deployment** | Hugging Face Spaces (API), Vercel (UI) |

---
## Features

- Dual Model Support — Switch between Q4 (fast) and Q8 (accurate) at runtime
- Streaming Responses — Real-time token streaming with Server-Sent Events
- OpenAI-Compatible API — Drop-in replacement for /v1/chat/completions
- API Key Auth — Simple Bearer token authentication
- Premium UI — Glass morphism, animated gradients, responsive design
- Medical Disclaimer — Ethical AI usage notice on every interaction

## Dual Support:

| Variable     | Value                     | Effect                  |
| ------------ | ------------------------- | ----------------------- |
| `MODEL_REPO` | `SatRag/chat-doctor-q4`   | Uses Q4 model           |
| `MODEL_FILE` | `chat-doctor-q4.gguf`     | Q4 filename             |
| `MODEL_REPO` | `SatRag/chat-doctor-gguf` | Uses Q8 model (default) |
| `MODEL_FILE` | `chat-doctor.gguf`        | Q8 filename             |

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

| Spec               | Value                                           |
| ------------------ | ----------------------------------------------- |
| Base Model         | `google/gemma-4-E2B-it`                         |
| Fine-tuning        | QLoRA (NF4) + LoRA adapter training             |
| Parameters         | 4B                                              |
| Quantizations      | Q4\_K\_M (~2.5GB), Q8\_0 (~5GB)                 |
| Context Length     | 2048 tokens                                     |
| Training Framework | TRL SFTTrainer with assistant-only masking      |
| LoRA Rank          | 8                                               |
| LoRA Alpha         | 16                                              |
| LoRA Dropout       | 0.05                                            |
| Training Hardware  | Kaggle NVIDIA T4                                |
| Training Time      | ~4–5 hours                                      |

---
## Training Data

ChatDoc was fine-tuned using a custom dataset created by combining:

- UltraChat conversational data
- MedDialog medical conversations

The combined dataset was cleaned, anonymized, converted to a unified
JSONL chat format, and split into:

- Training set (95%)
- Validation set (5%)

Dataset:
https://www.kaggle.com/datasets/satvikraghav/cleaned-anon-jsonl

Files:
- train.jsonl (1.38 GB)
- val_formatted.jsonl (73.25 MB)

---
## Model Lineage

```
Training Pipeline:

UltraChat + MedDialog
        ↓
Cleaned / Anonymized Dataset
        ↓
95/5 Train/Validation Split
        ↓
Fine-Tuned Gemma 4 E2B (QLoRA)
        ↓
LoRA Adapter

https://www.kaggle.com/datasets/satvikraghav/chat-doctor-gemma4-lora

        ↓ Merge with Base Model

https://www.kaggle.com/datasets/satvikraghav/chat-doctor-merged

        ↓ GGUF Conversion

Q8_0:
https://huggingface.co/SatRag/chat-doctor-gguf

Q4_K_M:
https://huggingface.co/SatRag/chat-doctor-q4
```

---
## Project Structure 

```
mavrk7-chatdoc/
├── api/            # FastAPI inference server
├── frontend/       # React/Vite web application
├── production/     # Training and deployment pipelines
├── research/       # Earlier transformer/MoE/distillation work
└── checkpoints/    # Model artifacts
```
For the complete repository structure, see the source tree.

---
## Env Variables


| Variable        | Default          | Description                        |
| --------------- | ---------------- | ---------------------------------- |
| `HF_TOKEN`      | —                | Hugging Face auth token            |
| `API_KEY`       | `test-key-123`   | API authentication key             |
| `DEFAULT_MODEL` | `chat-doctor-q4` | Default model on startup           |
| `MODEL_REPO`    | —                | Override HF repo for custom models |
| `MODEL_FILE`    | —                | Override GGUF filename             |


---

## Uptime Strategy

A lightweight uptime monitor (UptimeRobot) periodically sends HEAD requests to
the `/health` endpoint to reduce cold starts in serverless deployment.
---
## Limitations

- CPU-only inference — Responses take 5-15 seconds depending on length
- Not a substitute for professional medical advice — Always consult a doctor for serious symptoms
- General health Q&A only — Responses are informational and may be incorrect, incomplete, or outdated
- Not intended for diagnosis, treatment decisions, prescriptions, or emergency medical situations

---

## Project Evolution

ChatDoc began as an experiment in training a custom medical language model from scratch.

The project progressed through several stages:

1. Custom Transformer implementation
2. Mixture-of-Experts (MoE) experimentation
3. Knowledge distillation pipeline
4. Dataset curation and evaluation tooling
5. QLoRA fine-tuning of Gemma 4 E2B
6. GGUF quantization and production deployment

While the custom-model and distillation approaches did not achieve the desired quality-performance tradeoff, they provided valuable infrastructure for dataset processing, evaluation, and experimentation that ultimately informed the final ChatDoc system.
---
## Research Artifacts

The repository also contains earlier experimental work including:

- Custom Transformer implementations
- Mixture-of-Experts (MoE) architectures
- Distillation pipelines
- Dataset evaluation tooling
- Tokenizer experimentation

These artifacts are preserved for research and reproducibility purposes and document the evolution of the project before the final Gemma 4 QLoRA approach.

For more informaion about that please checkout the README in /research

---

## Acknowledgments

- llama.cpp for efficient CPU inference
- Hugging Face for model hosting and Spaces
- Google for the Gemma 4 model family