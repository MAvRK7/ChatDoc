import json
import torch
import sentencepiece as spm
from torch.utils.data import Dataset

IGNORE_INDEX = -100


class ConversationDataset(Dataset):
    def __init__(self, path, tokenizer_path, max_length=512):
        self.samples = []
        self.max_length = max_length

        # Load tokenizer properly
        self.tok = spm.SentencePieceProcessor()
        self.tok.load(tokenizer_path)

        # Get special token IDs from vocab (these are fixed now)
        self.pad_id = self.tok.piece_to_id("<pad>")      # 0
        self.unk_id = self.tok.piece_to_id("<unk>")      # 1 (control)
        self.bos_id = self.tok.piece_to_id("<bos>")      # 2
        self.eos_id = self.tok.piece_to_id("<eos>")      # 3
        self.user_id = self.tok.piece_to_id("<user>")    # 4
        self.assistant_id = self.tok.piece_to_id("<assistant>")  # 5

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")

                if "<user>" not in text or "<assistant>" not in text:
                    continue

                # Extract all turns
                turns = self._extract_turns(text)

                for user_text, assistant_text in turns:
                    input_ids = []
                    labels = []

                    # USER turn (no loss)
                    user_tokens = self._encode(f"<user> {user_text} ")
                    input_ids.extend(user_tokens)
                    labels.extend([IGNORE_INDEX] * len(user_tokens))

                    # ASSISTANT prompt (no loss)
                    assistant_prompt = self._encode("<assistant> ")
                    input_ids.extend(assistant_prompt)
                    labels.extend([IGNORE_INDEX] * len(assistant_prompt))

                    # ASSISTANT answer (train here)
                    answer_tokens = self._encode(assistant_text)
                    input_ids.extend(answer_tokens)
                    labels.extend(answer_tokens)

                    # Add EOS at end of assistant response
                    input_ids.append(self.eos_id)
                    labels.append(self.eos_id)

                    # truncate
                    input_ids = input_ids[:self.max_length]
                    labels = labels[:self.max_length]

                    self.samples.append((input_ids, labels))

    def _encode(self, text):
        """Encode text to IDs."""
        return self.tok.EncodeAsIds(text)

    def _extract_turns(self, text):
        """Extract (user_text, assistant_text) pairs from formatted text."""
        turns = []
        # Remove trailing <eos> only
        text = text.rstrip()
        if text.endswith("<eos>"):
            text = text[:-5].rstrip()

        # Split by <user> tag
        parts = text.split("<user>")

        for part in parts[1:]:  # skip before first <user>
            if "<assistant>" not in part:
                continue

            # Split into user part and assistant part
            split_idx = part.find("<assistant>")
            user_text = part[:split_idx].strip()
            rest = part[split_idx + len("<assistant>"):].strip()

            # If there's another <user> later, assistant text ends there
            next_user = rest.find("<user>")
            if next_user != -1:
                assistant_text = rest[:next_user].strip()
            else:
                assistant_text = rest.strip()

            if len(user_text) > 5 and len(assistant_text) > 10:
                turns.append((user_text, assistant_text))

        return turns

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_batch(batch, pad_id=0):
    max_len = max(len(x[0]) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

    for i, (ids, lbls) in enumerate(batch):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, :len(lbls)] = torch.tensor(lbls, dtype=torch.long)

    return input_ids, labels