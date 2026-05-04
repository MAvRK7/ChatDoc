# ChatDoc

A language model 

## 🧾 Quick facts:

- 120M parameters 
- MoE architecture (4 experts, top‑2 routing) [Equivalent to a 250M–300M dense model]
- 1024 context length
- Trains in 3–4 hours A100 GPU
- Fits in 4GB VRAM at inference
- Perfect for medical Q&A

Other details:

Has FlashAttention for speed, RMSNorm for stability, SwiGLU FFN, 20k BPE tokenizer

Total samples: 221318. Total tokens: 51,757,583. Average tokens per sample: 233.86. Vocab size: 20,000

## 📦 Dataset 

A custom dataset has been created for pre-training this model. It consists of:

- Ultra Chat: 185202 (54.03%)
- Smol Summarize: 101428 (29.59%)
- Smol rewrite: 56150 (16.38%)

Total dataset size: 342,780
    - Train size: 324,900 samples (95%)
    - Val size: 17,880 samples (5%)

---

📂 File Structure 

```text
chat-doctor/
│
├── data/                     # ignored
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── test.jsonl (for final test)
│   │   ├── english-train.json (train of MedDialogue)
│   │   ├── english-dev.json (val set of MedDialogue)
│   │   ├── HealthCareMagic-100k.json
│   │   └── medquad.csv
│   │
│   ├── processed/
│   │   ├── merged.jsonl
│   │   ├── train.jsonl (95% of merged.jsonl)
│   │   ├── val.jsonl (5%)
│   │   ├── healthcaremagic.jsonl
│   │   ├── meddialog_dev.jsonl
│   │   ├── medquad.jsonl
│   │   ├── raw_clean.jsonl
│   │   ├── combined_greetings_identity.jsonl
│   │   ├── adversarial.jsonl
│   │   └── mental_health
│   │
│   └── test/
│       ├── in_domain.jsonl (200 samples from test.csv)
│       ├── ood.jsonl
│       └── safety.jsonl
│
├── outputs/                 # initial random weights output
│   ├── runs/ #ignored
│   │   └──logs (tfevents file) used for TensorBoard
│   ├── in_domain_model_outputs
│   ├── ood_model_outputs.jsonl
│   └── safety_model_outputs.jsonl
│
├── src/
│   ├──__init__.py
│   ├── tokenizer.py/
│   │   ├── count_tokens.py
│   │   ├── train_tokenizer.py
│   │   ├── verify_tokenizer.py
│   │   ├── sample_token_corpus.py
│   │   ├── tokenizer.json.model
│   │   ├── tokenizer.json.vocab
│   │   ├── corpus.txt           # ignored
│   │   ├── sample_token_corpus.py
│   │   └── tokenizer_sampled.json_corpus.txt
│   │
│   ├── dataset/
│   │   ├── dataset.py
│   │   └── test_dataset.py
│   │
│   ├── model/
│   │   ├── moe.py
│   │   └── transformer.py
│   │
│   ├── scripts/
│   │   ├── convert_csv_to_jsonl.py
│   │   ├── convert_healthcaremagic.py
│   │   ├── convert_medquad.py
│   │   ├── dataset_cleaner.py
│   │   ├── gen_multi_geetings.py
│   │   ├── merge_datasets.py
│   │   ├── analyze_dataset.py
│   │   ├── split_cleaned_jsonl.py
│   │   ├── edge_cases/
│   │   │   ├── adversarial.py
│   │   │   └── mental_health.py
│   │   └── eval/
│   │       └──  eval_sets.py
│   │
│   ├── sampling.py
│   ├── inference.py
│   ├── train.py
│   ├── agent.py (WIP)
│   └── utils/ (WIP)
│       ├── config.py
│       └── logging.py
│
├── config.yaml (WIP)
├── tokenizer.json
├── requirements.txt
├── model.pt (the weights) # ignored
└── README.md

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
Details after Phase I pre-training
* Logits shape: torch.Size([2, 64, 8000])
* MoE loss: 0.4164277911186218