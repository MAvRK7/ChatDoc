# production/inference/inference.py
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_PATH = "checkpoints/gemma-merged"

print("Loading merged Gemma 4...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def chat(user_input, system_prompt=None, max_new_tokens=512):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})
    
    # Gemma 4 uses processor, not just tokenizer
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,      # Gemma 4 best practice
            top_p=0.95,
            top_k=64,
            do_sample=True,
        )
    
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    
    # Parse thinking if present
    if "<|channel>" in response:
        # Extract final answer after thinking
        parts = response.split("<channel|>")
        if len(parts) > 1:
            response = parts[-1]
    
    # Clean up
    response = response.replace("<|turn|>", "").replace("<turn|>", "").strip()
    
    return response


if __name__ == "__main__":
    # Test
    print("User: Explain quantum computing simply")
    print(f"Assistant: {chat('Explain quantum computing simply')}")
    
    print("\nUser: I have a headache and fever, what should I do?")
    print(f"Assistant: {chat('I have a headache and fever, what should I do?')}")