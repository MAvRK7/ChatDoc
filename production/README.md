Production-ready fine-tuning and inference pipeline for the local assistant.

This directory contains the deployable system built around:

- Gemma 4 E2B
- QLoRA fine-tuning
- Synthetic instruction distillation
- Local inference with FastAPI or vLLM

The original custom transformer + MoE experiments are preserved separately in /research.

Pipeline

```text
Teacher Generation
    ↓
Dataset Conversion
    ↓
QLoRA Fine-Tuning
    ↓
LoRA Merge
    ↓
Local Inference
```

Hardware
Optimized for:
- Kaggle T4 GPUs
- Consumer NVIDIA GPUs
- fp16 + 4-bit QLoRA