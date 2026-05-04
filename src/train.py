import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch import amp
import sentencepiece as spm

from torch.utils.tensorboard import SummaryWriter

from src.dataset.dataset import ConversationDataset, collate_batch
from src.model.transformer import MoETransformer


class TrainConfig:
    repo_root = os.getcwd()

    train_path = os.path.join(repo_root, "data/processed/train_formatted.jsonl")
    val_path = os.path.join(repo_root, "data/processed/val_formatted.jsonl")
    tokenizer_path = os.path.join(repo_root, "tokenizer/tokenizer.model")  
    #save_path = os.path.join(repo_root, "checkpoints/model.pt")
    save_path = os.path.join(repo_root, "checkpoints", "model.pt")

    log_dir = os.path.join(repo_root, "outputs/runs")

    batch_size = 16
    grad_accum_steps = 4
    max_length = 1024

    lr = 1e-4
    weight_decay = 0.01

    warmup_steps = 300
    max_steps = 30000

    eval_every = 1000
    log_every = 100

    early_stopping_patience = 5
    early_stopping_min_delta = 0.01

    device = "cuda" if torch.cuda.is_available() else "cpu"


def cosine_lr(step, max_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    p = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


def evaluate(model, dl, device, vocab_size, pad_id):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0
    count = 0

    with torch.no_grad(), amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        for batch, labels in dl:
            batch = batch.to(device)
            labels = labels.to(device)

            logits, _ = model(batch)

            logits = logits[:, :-1].reshape(-1, vocab_size)
            labels = labels[:, 1:].reshape(-1)

            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.numel()
            count += labels.numel()

    model.train()
    return total_loss / max(count, 1)


def generate(model, tok, prompt, device, max_new_tokens=80):
    """Generation with proper tokenizer API."""
    model.eval()

    # Get IDs from tokenizer
    eos_id = tok.piece_to_id("<eos>")
    bos_id = tok.piece_to_id("<bos>")
    
    # Encode prompt
    ids = tok.EncodeAsIds(prompt)
    
    # Optionally prepend BOS if not present
    if len(ids) == 0 or ids[0] != bos_id:
        ids = [bos_id] + ids
        
    x = torch.tensor([ids], device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x)
            logits = logits[:, -1, :] / 0.8

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)

            if next_id.item() == eos_id:
                break

            x = torch.cat([x, next_id], dim=1)

    # Decode IDs back to text
    out_ids = x[0].tolist()
    out = tok.DecodeIds(out_ids)
    model.train()
    return out


def train():
    cfg = TrainConfig()

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    writer = SummaryWriter(cfg.log_dir)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(cfg.tokenizer_path)
    vocab_size = sp.get_piece_size()
    
    # Get special token IDs
    pad_id = sp.piece_to_id("<pad>")
    eos_id = sp.piece_to_id("<eos>")
    bos_id = sp.piece_to_id("<bos>")

    # Pass tokenizer_path instead of tokenizer object
    train_ds = ConversationDataset(cfg.train_path, cfg.tokenizer_path, cfg.max_length)
    val_ds = ConversationDataset(cfg.val_path, cfg.tokenizer_path, cfg.max_length)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=lambda x: collate_batch(x, pad_id=pad_id))
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=lambda x: collate_batch(x, pad_id=pad_id))

    model = MoETransformer(
        vocab_size=vocab_size,
        dim=768,
        num_layers=12,
        num_heads=12,
        ffn_hidden_dim=2048,
        num_experts=4,
        k=2,
        max_seq_len=cfg.max_length,
    ).to(cfg.device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = amp.GradScaler(enabled=(cfg.device == "cuda"))
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    start_step = 0

    if os.path.exists(cfg.save_path):
        try:
            print("Loading checkpoint...")
            checkpoint = torch.load(cfg.save_path, map_location=cfg.device)

            model.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["optimizer"])
            start_step = checkpoint.get("step", 0)
            print(f"Resumed from step {start_step}")

        except Exception as e:
            print(f"Checkpoint load failed: {e}")
            print("Starting fresh training.")
    
    else:
        print("Checkpoint not available.")

    step = start_step
    best_val_loss = float("inf")
    no_improve_steps = 0
    opt.zero_grad()

    model.train()

    while step < cfg.max_steps:
        for batch, labels in train_dl:
            batch = batch.to(cfg.device)
            labels = labels.to(cfg.device)

            with amp.autocast(device_type="cuda", enabled=(cfg.device == "cuda")):
                logits, aux = model(batch)
            
                logits_flat = logits[:, :-1].reshape(-1, vocab_size)
                labels_flat = labels[:, 1:].reshape(-1)
            
                ce_loss = loss_fn(logits_flat, labels_flat)

                if isinstance(aux, dict):
                    moe_loss = aux.get("moe_loss", torch.tensor(0.0, device=cfg.device))
                else:
                    moe_loss = aux if torch.is_tensor(aux) else torch.tensor(0.0, device=cfg.device)

                loss = ce_loss + 0.01 * moe_loss

            scaler.scale(loss).backward()

            if step % 200 == 0:
                with torch.no_grad():
                    preds = torch.argmax(logits_flat, dim=-1)
                    mask = labels_flat != -100
                    correct = (preds[mask] == labels_flat[mask]).float().mean()
                    print(f"[DEBUG] token_acc={correct.item():.4f}")
                    writer.add_scalar("train/token_acc", correct.item(), step)
            
            # Add this temporarily in train.py inside the training loop DEBUG 2
            if step == 500:
                print("Label stats:", (labels != -100).sum().item(), "trainable out of", labels.numel())
                print("First batch labels:", labels[0, :20])

            if isinstance(aux, dict) and "gate_scores" in aux and len(aux["gate_scores"]) > 0:
                gs = aux["gate_scores"][0]
                expert_usage = gs.mean(dim=(0, 1))
                entropy = -(expert_usage * torch.log(expert_usage + 1e-9)).sum()
                writer.add_scalar("moe/entropy", entropy.item(), step)
                for i, usage in enumerate(expert_usage):
                    writer.add_scalar(f"moe/expert_{i}", usage.item(), step)

            if (step + 1) % cfg.grad_accum_steps == 0:
                lr = cosine_lr(step, cfg.max_steps, cfg.lr, cfg.warmup_steps)
                for g in opt.param_groups:
                    g["lr"] = lr

                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

                writer.add_scalar("train/lr", lr, step)

            if step % cfg.log_every == 0:
                current_loss = (ce_loss + 0.01 * moe_loss).detach().item()
                current_lr = opt.param_groups[0]["lr"]

                writer.add_scalar("train/loss", current_loss, step)
                writer.add_scalar("train/ce_loss", ce_loss.item(), step)
                writer.add_scalar("train/moe_loss", moe_loss.detach().item() if torch.is_tensor(moe_loss) else 0.0, step)

                ppl = math.exp(min(ce_loss.item(), 15))
                writer.add_scalar("train/ppl", ppl, step)

                print(
                    f"[STEP {step}] "
                    f"loss={current_loss:.4f} | "
                    f"ce={ce_loss.item():.4f} | "
                    f"moe={moe_loss.item() if torch.is_tensor(moe_loss) else 0.0:.4f} | "
                    f"lr={current_lr:.2e} | "
                    f"ppl={ppl:.2f}"
                )

            if step % cfg.eval_every == 0 and step > 0:
                val_loss = evaluate(model, val_dl, cfg.device, vocab_size, pad_id)
                print(f"[VAL] {step} loss={val_loss:.4f}")
                writer.add_scalar("val/loss", val_loss, step)

                if val_loss < best_val_loss - cfg.early_stopping_min_delta:
                    best_val_loss = val_loss
                    no_improve_steps = 0
                    print("[INFO] New best model — saving")
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "step": step
                    }, cfg.save_path)
                else:
                    no_improve_steps += 1
                    print(f"[INFO] No improvement ({no_improve_steps}/{cfg.early_stopping_patience})")
                    if no_improve_steps >= cfg.early_stopping_patience:
                        print("Early stopping triggered.")
                        return

                # FIXED: Pass tokenizer object properly
                sample = generate(
                    model, sp,
                    "<user> I have a headache. What can I do? <assistant> ",
                    cfg.device
                )
                print("[GEN]", sample)

            step += 1
            if step >= cfg.max_steps:
                break

    print("Training complete")


if __name__ == "__main__":
    train()

# run 
# python -m src.train