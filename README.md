# Aura-MoE

A Sparse Mixture-of-Experts Architecture from Scratch

## 🧾 Quick facts:

- Parameters: 138M (Architecture equivalent to a 250M–300M dense model)
- Architecture: 12-layer Transformer with Interleaved Dense and MoE blocks.
- MoE Config: 4 Experts with Top-1 Routing (k=1).
- Context Length: 1024 token
- Efficiency: Fits in 4GB VRAM at inference; trains in ~4 hours on an A100
- Tokenizer: Custom 20k vocabulary SentencePiece model.
- Uses RoPE (Rotary Positional Embeddings) and SwiGLU activators.

Aura-MoE is a 138M parameter Transformer model implementing a sparse Mixture-of-Experts (MoE) layer. This project serves as a deep-dive into the engineering challenges of training MoE architectures, featuring custom implementations of modern LLM components like RoPE, SwiGLU, and FlashAttention.

Other details:

Has FlashAttention for speed, RMSNorm for stability, 20k sentancepiece tokenizer

| Metric                              | Value        |
|-------------------------------------|--------------|
| Total samples (turn-level)          | 710,647      |
| Total tokens (after truncation)     | 234,781,772  |
| Trainable tokens (assistant only)   | 165,057,619  |
| Average tokens per sample           | 330.38       |
| Trainable ratio                     | 70.30%       |
| Vocab size                          | 20,000       |

---

## 🏗️ Technical Architecture

The model is built with a decoupled, modular design to allow for rapid experimentation with routing and expert scaling.  

- Parameter Count: 138,534,912  

- Layers: 12-layer Transformer (Interleaved Dense and MoE blocks)  

- MoE Config: 4 Experts, Top-1 Routing (k=1)  

- Hidden Dimension: 768  

- Context Window: 1024 token

### Key Engineering Features

* Custom MoE Layer: Features a manual dispatch/combine logic with auxiliary load-balancing and Z-loss for gating stability.  

* Rotary Positional Embeddings (RoPE): Implemented from scratch for better long-context performance.  

* Optimized Inference: Includes a dedicated sampling engine with n-gram blocking and repetition penalty to manage semantic drift.  

* Memory Efficiency: Built with torch.amp for mixed-precision training and gradient accumulation for larger effective batch sizes.

## 📦 Dataset 

A custom dataset has been created for pre-training this model. It consists of:

| Dataset          | Samples | Percentage | Approx. Tokens |
|------------------|---------|------------|----------------|
| Ultra Chat       | 185,202 | 54.03%     | ~126M          |
| Smol Summarize   | 101,428 | 29.59%     | ~69M           |
| Smol Rewrite     | 56,150  | 16.38%     | ~39M           |

Total dataset size: 342,780
    - Train size: 324,900 samples (95%)
    - Val size: 17,880 samples (5%)

---

📂 File Structure 

```text

├── src/
│   ├── model/
│   │   ├── transformer.py  # Core MoE Transformer & RoPE logic
│   │   └── moe.py          # Expert dispatch & gating mechanism
│   ├── dataset/            # Formatting & tokenization pipelines
│   ├── engine/             # Training & Evaluation loops
│   └── inference.py        # Generation with repetition penalties
├── tokenizer/              # Custom SentencePiece model
└── outputs/                # Tensorboard logs & OOD benchmarks

```
---

## 🛠️ Installation

### 1️⃣ Clone the repository

```git 
git clone https://github.com/MAvRK7/chat-doctor.git
cd chat-doctor
```
### 2️⃣ Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```
### 4️⃣ Dataset Setup


1. Download the dataset

Download train.csv manually from:

[train dataset](https://www.kaggle.com/datasets/satvikraghav/cleaned-anon-jsonl/data?select=train.jsonl)

[validation dataset](https://www.kaggle.com/datasets/satvikraghav/cleaned-anon-jsonl/data?select=val.jsonl)

2. Place the file

Move the downloaded train file to:

```
data/processed/train.jsonl
```

and place the val dataset in 

```
data/processed/val.jsonl
```

The split in the train dataset (train.jsonl) into train (95%) and validation (val.jsonl) is a 95/5 split 


⚠️ Note: All the # ignored tagged files are ignored in Git due to file size limits, so you must download the dataset locally before running the project.


---

The Pretraining phase I model weights are available at 

[model.pt](https://www.kaggle.com/datasets/satvikraghav/chat-doctor-checkpoints/data)

Once trained, it should be placed in the project root.

---

Training details on TensorBoard

Place the tfevents in 

```
outputs/runs
```

Run

```
cd /Users/satvikraghav/coding/chat-doctor
tensorboard --logdir outputs/runs
```

This will generate a link
```
http://localhost:6006/
```
on which the trainig charts will be visible. Available charts are:

Train:

train/ce_loss, train/loss, train/lr, train/moe_loss

Val:

val/loss, val and val/perplexity

The text generated during trining stages will also be visible in the Text section.
---

Note:

Due to compute constraints (30k steps on ~0.5B tokens), the current checkpoint demonstrates high syntactic fluency and assistant-persona alignment but remains in the "early-convergence" phase regarding factual grounding

Sample Output

User: Explain AI in education?

Assistant: The first step in the future of your life was to have a significant impact on our lives. Its life has been a crucial part of you, and it has had a positive impact on you. In this time, I have made you feel more confident in my life. It's a great way to focus on the life that we can do, and that we are all about the most importantly in our lives and how we can make a difference in your life. Let me know if you have any questions or need it?