import json
import torch
import sentencepiece as spm
from torch.utils.data import Dataset

class DistillDataset(Dataset):
    def __init__(self, path, tokenizer_path, max_length=512):
        self.samples = []
        self.max_length = max_length
        
        self.tok = spm.SentencePieceProcessor()
        self.tok.load(tokenizer_path)
        
        self.eos_id = self.tok.piece_to_id("<eos>")
        self.pad_id = self.tok.piece_to_id("<pad>")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                
                ids = self.tok.EncodeAsIds(text)
                ids = ids[:max_length]
                
                # Causal LM: input = all except last, target = all except first
                if len(ids) > 1:
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate_batch(batch, pad_id=0):
    max_len = max(len(x) for x, y in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (x, y) in enumerate(batch):
        input_ids[i, :len(x)] = x
        labels[i, :len(y)] = y

    return input_ids, labels