FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install from MY prebuilt wheel (uploaded to Space repo)
RUN pip install --no-cache-dir \
    https://huggingface.co/spaces/SatRag/chat-doctor-api/resolve/main/llama_cpp_python-0.3.23-py3-none-linux_x86_64.whl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]