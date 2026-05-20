from huggingface_hub import login
import os

token = os.getenv("HF_TOKEN")
if token:
    login(token)

import torch
from threading import Thread
import uuid
import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

from peft import PeftModel

from config import (
    BASE_MODEL,
    ADAPTER_REPO,
    SYSTEM_PROMPT,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    REPETITION_PENALTY,
)

_model = None
_tokenizer = None

# =========================================================
# Detect device automatically
# =========================================================
HAS_GPU = torch.cuda.is_available()

if HAS_GPU:
    DEVICE = "cuda"
    print("GPU detected — using 4-bit quantization")
else:
    DEVICE = "cpu"
    print("No GPU detected — using CPU mode")

# =========================================================
# Gemma4 ClippableLinear fix
# =========================================================
def unwrap_clippable_linears(model):
    try:
        from transformers.models.gemma4 import modeling_gemma4

        ClippableLinear = getattr(
            modeling_gemma4,
            "Gemma4ClippableLinear",
            None,
        )

        if ClippableLinear:
            for _, parent in list(model.named_modules()):
                for child_name, child in list(parent.named_children()):
                    if isinstance(child, ClippableLinear):
                        setattr(parent, child_name, child.linear)

    except Exception as e:
        print(f"unwrap warning: {e}")

    return model

# =========================================================
# Load model
# =========================================================
def load_model():
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    print("Loading tokenizer...")

    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    print("Loading base model...")

    # =====================================================
    # GPU PATH (4-bit quantization)
    # =====================================================
    if HAS_GPU:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    # =====================================================
    # CPU PATH
    # =====================================================
    else:

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
            trust_remote_code=True,
        )

    print("Applying Gemma unwrap fix...")
    base = unwrap_clippable_linears(base)

    print("Loading LoRA adapter...")

    _model = PeftModel.from_pretrained(
        base,
        ADAPTER_REPO,
    )

    _model.eval()

    print("Model ready!")

    return _model, _tokenizer

# =========================================================
# Generate response
# =========================================================
def generate_response(
    messages,
    max_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    stream=False,
):

    model, tokenizer = load_model()

    # Add system prompt if missing
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(
            0,
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
        )

    # Chat formatting
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    # Move tensors safely
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    # =====================================================
    # STREAMING
    # =====================================================
    if stream:

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
        )

        thread = Thread(
            target=model.generate,
            kwargs=generation_kwargs,
        )

        thread.start()

        return streamer

    # =====================================================
    # NORMAL GENERATION
    # =====================================================
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]

    response_text = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "chat-doctor-v1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text.strip(),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_length,
            "completion_tokens": len(outputs[0]) - input_length,
            "total_tokens": len(outputs[0]),
        },
    }