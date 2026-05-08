import json
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/tokenizer.model")

with open("data/teacher_responses.jsonl") as f_in, \
     open("data/distill_train.jsonl", "w") as f_out:
    
    count = 0
    skipped = 0
    
    for line in f_in:
        item = json.loads(line)
        prompt = item["prompt"].strip()
        response = item["response"].strip()
        
        # Format like your training data
        text = f"{prompt}{response}\n<eos>"
        
        # Verify round-trip
        ids = sp.EncodeAsIds(text)
        decoded = sp.DecodeIds(ids)
        
        if text.rstrip() == decoded.rstrip():
            json.dump({"text": text}, f_out)
            f_out.write("\n")
            count += 1
        else:
            skipped += 1
    
    print(f"Converted: {count}, Skipped (round-trip failed): {skipped}")