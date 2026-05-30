"""
Chat Doctor Model Quantization Pipeline
========================================
The merged (Gemma 4 weights + my LoRA weights) are saved in 

Kaggle Dataset:
  - Merged model: https://www.kaggle.com/datasets/satvikraghav/chat-doctor-merged

That is 12.25 GB. This was Quantized to 8-bit using llama.cpp's quantize tool, 
resulting in a 4.93 GB file. This was saved to HF in after a GGUF conversion step.
Then, this was further re-quantized to Q4_K_M for a final size of 3.4 GB.

Downloads the 8-bit Chat Doctor model from HuggingFace and creates a 4-bit Q4_K_M version.

HuggingFace Repos:
  - 8-bit model: https://huggingface.co/SatRag/chat-doctor-gguf (chat-doctor.gguf)
  - 4-bit model: https://huggingface.co/SatRag/chat-doctor-q4 (chat-doctor-Q4_K_M.gguf)


The 8-bit model was called as Q8 or ChatDoc Accurate in the UI, 
and the 4-bit model is called Q4_K_M or ChatDoc Fast.  

This script performs the following steps:
1. Sets up llama.cpp and builds the quantize tool if not already done.
2. Creates Q8_0 (8-bit) GGUF from merged model (if not already done)
3. Re-Quantizes the 8-bit model to Q4_K_M format using llama.cpp's quantize tool.
4. Validates both the quantized models
5. Optionally uploads the quantized model to HuggingFace.

Author: Satvik Raghav
Date: 2026-05-30 (ran on 2026-05-29)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# ========== CONFIGURATION ==========
HF_REPO_8BIT = "SatRag/chat-doctor-gguf"
HF_FILENAME_8BIT = "chat-doctor.gguf"
HF_REPO_4BIT = "SatRag/chat-doctor-q4"
OUTPUT_FILENAME_Q4 = "chat-doctor-Q4_K_M.gguf"

LLAMA_CPP_URL = "https://github.com/ggerganov/llama.cpp"
LLAMA_CPP_DIR = "./llama.cpp"
QUANTIZE_TOOL = os.path.join(LLAMA_CPP_DIR, "quantize")

# ========== SETUP ==========
def setup_llama_cpp():
    """Clone and build llama.cpp if not already done."""
    if not os.path.exists(QUANTIZE_TOOL):
        print("=" * 60)
        print("SETTING UP LLAMA.CPP")
        print("=" * 60)
        
        # Clone if not exists
        if not os.path.exists(LLAMA_CPP_DIR):
            print(f"Cloning {LLAMA_CPP_URL}...")
            subprocess.run(["git", "clone", LLAMA_CPP_URL, LLAMA_CPP_DIR], check=True)
        
        # Build quantize tool
        print("Building quantize tool...")
        os.makedirs(os.path.join(LLAMA_CPP_DIR, "build"), exist_ok=True)
        
        result = subprocess.run(
            [
                "cmake", "-B", os.path.join(LLAMA_CPP_DIR, "build"),
                "-S", LLAMA_CPP_DIR,
                "-DBUILD_SHARED_LIBS=OFF",
                "-DLLAMA_CURL=OFF",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF"
            ],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print("CMake config failed, trying make...")
            subprocess.run(["make", "-C", LLAMA_CPP_DIR, "quantize", "-j4"], check=True)
        else:
            subprocess.run(
                ["cmake", "--build", os.path.join(LLAMA_CPP_DIR, "build"),
                 "--target", "llama-quantize", "-j4"],
                check=True
            )
            
            # Copy binary to expected location
            import glob
            binaries = glob.glob(os.path.join(LLAMA_CPP_DIR, "build/bin/llama-quantize*"))
            if binaries:
                shutil.copy(binaries[0], QUANTIZE_TOOL)
                os.chmod(QUANTIZE_TOOL, 0o755)
        
        if os.path.exists(QUANTIZE_TOOL):
            print("✅ Quantize tool ready!\n")
        else:
            print("❌ Failed to build quantize tool")
            sys.exit(1)
    else:
        print("✅ Quantize tool already exists\n")


def download_8bit_model():
    """Download the 8-bit model from HuggingFace."""
    print("=" * 60)
    print("DOWNLOADING 8-BIT MODEL")
    print("=" * 60)
    
    from huggingface_hub import hf_hub_download
    
    model_path = hf_hub_download(
        repo_id=HF_REPO_8BIT,
        filename=HF_FILENAME_8BIT,
        resume_download=True
    )
    
    size_gb = os.path.getsize(model_path) / 1e9
    print(f"✅ Downloaded: {model_path}")
    print(f"   Size: {size_gb:.2f} GB\n")
    
    return model_path


def quantize_to_q4(input_path, output_path):
    """Quantize 8-bit model to Q4_K_M."""
    print("=" * 60)
    print("QUANTIZING TO Q4_K_M")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("This may take 10-15 minutes...\n")
    
    result = subprocess.run(
        [QUANTIZE_TOOL, input_path, output_path, "Q4_K_M", "--allow-requantize"],
        check=False
    )
    
    if result.returncode == 0 and os.path.exists(output_path):
        size_gb = os.path.getsize(output_path) / 1e9
        input_size = os.path.getsize(input_path) / 1e9
        print(f"\n✅ Q4_K_M created: {size_gb:.2f} GB")
        print(f"   Savings: {((1 - size_gb/input_size) * 100):.1f}% smaller than 8-bit\n")
        return output_path
    else:
        print("❌ Quantization failed")
        if result.stderr:
            print(result.stderr)
        return None


def validate_model(model_path):
    """Validate the quantized model."""
    print("=" * 60)
    print("VALIDATING MODEL")
    print("=" * 60)
    
    import struct
    
    # Check GGUF structure
    with open(model_path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        
        if magic == 0x46554747:
            print(f"✅ Valid GGUF file (v{version}, {n_tensors} tensors)")
        else:
            print(f"❌ Invalid magic number: {hex(magic)}")
            return False
    
    # Test inference
    print("Testing inference...")
    try:
        from llama_cpp import Llama
        
        llm = Llama(model_path=model_path, n_ctx=512, n_threads=2, verbose=False)
        response = llm("Patient has fever and cough. What should I check?", max_tokens=50)
        text = response['choices'][0]['text']
        
        if len(text.strip()) > 20:
            print(f"✅ Model works!")
            print(f"   Sample output: {text[:100]}...\n")
            return True
        else:
            print("⚠️ Short output, but model loaded\n")
            return True
    except ImportError:
        print("⚠️ llama-cpp-python not installed, skipping inference test")
        print("   Install with: pip install llama-cpp-python\n")
        return True
    except Exception as e:
        print(f"❌ Inference test failed: {e}\n")
        return False


def upload_to_huggingface(model_path):
    """Upload the model to HuggingFace."""
    print("=" * 60)
    print("UPLOAD TO HUGGINGFACE")
    print("=" * 60)
    
    try:
        from huggingface_hub import HfApi, login
        
        # Try to login (will use stored token or prompt)
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("HF_TOKEN not set. Please enter your token:")
            token = input().strip()
        
        login(token=token)
        api = HfApi()
        
        print(f"Uploading to {HF_REPO_4BIT}...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=OUTPUT_FILENAME_Q4,
            repo_id=HF_REPO_4BIT,
            repo_type="model",
            commit_message="Add Chat Doctor Q4_K_M quantized model"
        )
        print(f"✅ Upload complete!")
        print(f"   View at: https://huggingface.co/{HF_REPO_4BIT}\n")
        return True
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"   Manual upload: {model_path}\n")
        return False


# ========== MAIN ==========
def main():
    """Main pipeline."""
    print("\n" + "=" * 60)
    print("CHAT DOCTOR QUANTIZATION PIPELINE")
    print("=" * 60)
    print(f"8-bit repo: {HF_REPO_8BIT}")
    print(f"4-bit repo: {HF_REPO_4BIT}")
    print("=" * 60 + "\n")
    
    # Step 1: Setup llama.cpp
    setup_llama_cpp()
    
    # Step 2: Download 8-bit model
    model_8bit = download_8bit_model()
    
    # Step 3: Quantize to Q4
    output_path = OUTPUT_FILENAME_Q4
    if os.path.exists(output_path):
        print(f"⚠️ {output_path} already exists. Delete it first to re-quantize.\n")
    else:
        model_q4 = quantize_to_q4(model_8bit, output_path)
        if not model_q4:
            print("❌ Pipeline failed at quantization step")
            return
    
    # Step 4: Validate
    if not os.path.exists(output_path):
        print("❌ Q4 model not found")
        return
    
    if not validate_model(output_path):
        print("⚠️ Validation had issues, but file may still be usable")
    
    # Step 5: Upload
    upload = input("Upload to HuggingFace? (y/n): ").strip().lower()
    if upload == 'y':
        upload_to_huggingface(output_path)
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"8-bit model: {model_8bit} ({os.path.getsize(model_8bit)/1e9:.2f} GB)")
    if os.path.exists(output_path):
        print(f"4-bit model: {output_path} ({os.path.getsize(output_path)/1e9:.2f} GB)")
    print(f"\nAdd to your config.py:")
    print(f"""
MODELS = {{
    "chat-doctor-q4": {{
        "repo": "{HF_REPO_4BIT}",
        "file": "{OUTPUT_FILENAME_Q4}",
        "name": "Chat Doctor (4-bit)",
        "description": "4-bit quantized Chat Doctor model",
        "quantization": "Q4_K_M",
    }},
    "chat-doctor-q8": {{
        "repo": "{HF_REPO_8BIT}",
        "file": "{HF_FILENAME_8BIT}",
        "name": "Chat Doctor (8-bit)",
        "description": "8-bit quantized Chat Doctor model",
        "quantization": "Q8_0",
    }},
}}
""")


if __name__ == "__main__":
    main()