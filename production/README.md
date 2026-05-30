Production-ready fine-tuning and inference pipeline for a medical chatbot system.

This directory contains the deployable system built around:

- Gemma 4 E2B
- QLoRA fine-tuning
- Synthetic instruction distillation
- Local inference with FastAPI or vLLM

The original custom transformer + MoE experiments are preserved separately in /research.
---
Pipeline

```text
Dataset Cleaning & Formatting
    ↓
Teacher Generation (Distillation / Synthetic Data)
    ↓
Dataset Conversion
    ↓
QLoRA Fine-Tuning (Gemma 4 E2B)
    ↓
LoRA Merge
    ↓
GGUF Conversion & Quantization (Q8 / Q4)
    ↓
Local Inference (FastAPI / llama.cpp)
```
---
Hardware

Optimized for:
- Kaggle NVIDIA T4 GPUs (training)
- Consumer NVIDIA GPUs (inference)
- 4-bit QLoRA (NF4) during training
- GGUF quantization (Q8/Q4) for deployment
---
Note:
This directory contains only production-grade training and inference code.
Experimental architectures (custom Transformer, MoE, distillation research)
are located in `/research`.