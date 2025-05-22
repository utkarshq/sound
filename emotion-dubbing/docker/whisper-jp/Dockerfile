FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3 \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and related packages
RUN pip3 install --no-cache-dir \
    "numpy<2.0" \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    pyyaml \
    numpy 

# Install Whisper and pre-download the model
RUN pip3 install -U openai-whisper && \
    python3 -c "import whisper; whisper.load_model('large-v3')"

# Set proper PATH and create model cache directory
ENV PATH="/usr/local/bin:/root/.local/bin:/usr/local/cuda/bin:${PATH}"
ENV XDG_CACHE_HOME="/root/.cache"

# Create a volume for the model cache
VOLUME /root/.cache/whisper

# Verify installations
RUN python3 -c "import numpy; import torch; print(f'NumPy version: {numpy.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
RUN which whisper

# Set the working directory
WORKDIR /app

# Copy our wrapper script
COPY docker/whisper-jp/scripts/whisper_wrapper.py /app/
RUN chmod +x /app/whisper_wrapper.py

# Default command
ENTRYPOINT ["python3", "/app/whisper_wrapper.py"]