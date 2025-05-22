# Use NVIDIA PyTorch container with CUDA 12.2.2 (compatible with CUDA 12.1)
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Fix protobuf and ONNX issues for NeMo
RUN pip install --no-cache-dir \
    'protobuf<=3.20.0' \
    'onnx<=1.12.0'

# Install compatible versions of Hugging Face and related tools
RUN pip install --no-cache-dir \
    'huggingface-hub==0.21.3' \
    'transformers==4.40.0'

# Install transformer-engine
RUN pip install --no-cache-dir transformer-engine==1.13.0

# Install grpcio separately
RUN pip install --no-cache-dir grpcio==1.67.0

# Install NeMo toolkit dependencies
RUN pip install --no-cache-dir \
    'hydra-core==1.3.2' \
    'omegaconf>=2.3.0'

# Install NeMo toolkit with dependencies
RUN pip install --no-cache-dir \
    'nemo_toolkit[asr,tts,nlp]==1.20.0' \
    'torch==2.0.1' \
    'torchaudio==2.0.2' \
    'pytorch-lightning==2.0.9'

# Create app directory
WORKDIR /app
COPY . .

# Set environment variables for better performance
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set ulimits for better performance
RUN ulimit -n 64000
RUN ulimit -s 65536
