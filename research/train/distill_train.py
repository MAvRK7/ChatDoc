import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch import amp
import sentencepiece as spm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.dataset.distill_dataset import DistillDataset, collate_batch
from research.model.transformer import MoETransformer

class Config:
    data_path = "data/distill_train.jsonl"
    tokenizer_path = "tokenizer/tokenizer.model"
    save_path = "checkpoints/distill_model.pt"
    log_dir = "outputs/distill_runs"
    
    batch_size = 8          # per GPU
    grad_accum_steps = 4    # effective batch = 8 * 2 GPUs * 4 = 64
    max_length = 512
    lr = 5e-5               # Lower LR for fine-tuning
    weight_decay = 0.01
    warmup_steps = 200
    max_steps = 10000
    eval_every = 1000
    log_every = 50

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    cfg = Config()
    device = torch.device(f"cuda:{rank}")
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(cfg.tokenizer_path)
    vocab_size = sp.get_piece_size()
    pad_id = sp.piece_to_id("<pad>")
    
    # Dataset with DDP sampler
    dataset = DistillDataset(cfg.data_path, cfg.tokenizer_path, cfg.max_length)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler,
        collate_fn=lambda x: collate_batch(x, pad_id=pad_id),
        num_workers=2, pin_memory=True
    )
    
    # Model
    model = MoETransformer(
        vocab_size=vocab_size,
        dim=512,
        num_layers=8,
        num_heads=8,
        ffn_hidden_dim=1536,
        num_experts=2,
        k=1,
        max_seq_len=cfg.max_length,
    ).to(device)
    
    # Load pretrained weights
    pretrained_path = "checkpoints/model.pt"
    if os.path.exists(pretrained_path) and rank == 0:
        print("Loading pretrained weights...")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=False)
        print("Loaded pretrained weights")
    
    # Sync across GPUs
    if world_size > 1:
        dist.barrier()
    
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(enabled=True)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def get_lr(step):
        if step < cfg.warmup_steps:
            return cfg.lr * step / cfg.warmup_steps
        p = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * p))
    
    if rank == 0:
        writer = SummaryWriter(cfg.log_dir)
        os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    
    step = 0
    model.train()
    opt.zero_grad()
    
    while step < cfg.max_steps:
        sampler.set_epoch(step)
        
        for batch, labels in dataloader:
            batch = batch.to(device)
            labels = labels.to(device)
            
            with amp.autocast(device_type="cuda", enabled=True):
                logits, aux = model(batch)
                
                logits_flat = logits[:, :-1].reshape(-1, vocab_size)
                labels_flat = labels[:, 1:].reshape(-1)
                
                ce_loss = loss_fn(logits_flat, labels_flat)
                
                if isinstance(aux, dict):
                    moe_loss = aux.get("moe_loss", torch.tensor(0.0, device=device))
                else:
                    moe_loss = torch.tensor(0.0, device=device)
                
                loss = ce_loss + 0.01 * moe_loss
            
            scaler.scale(loss).backward()
            
            if (step + 1) % cfg.grad_accum_steps == 0:
                lr = get_lr(step)
                for g in opt.param_groups:
                    g["lr"] = lr
                
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            
            if rank == 0 and step % cfg.log_every == 0:
                ppl = math.exp(min(ce_loss.item(), 15))
                print(f"[STEP {step}] loss={loss.item():.4f} | ce={ce_loss.item():.4f} | ppl={ppl:.2f}")
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/ppl", ppl, step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], step)
            
            step += 1
            if step >= cfg.max_steps:
                break
        
        if rank == 0 and step % cfg.eval_every == 0 and step > 0:
            torch.save({
                "model": model.module.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step
            }, cfg.save_path)
            print(f"[INFO] Saved checkpoint at step {step}")
    
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    print(f"Starting DDP training on {world_size} GPUs")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()