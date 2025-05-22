FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git cmake g++ make \
    libopenblas-dev portaudio19-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Fix potential permission issues
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install OpenSMILE from source with proper configuration
RUN git clone --depth 1 --branch v3.0.0 https://github.com/audeering/opensmile.git && \
    cd opensmile && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_ALL=ON -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /opensmile

# Create proper config directory
RUN mkdir -p /usr/local/share/opensmile/config

# Copy requirements and install Python dependencies
WORKDIR /app
COPY docker/emotion-features/requirements.txt .
RUN pip install -r requirements.txt

# Copy scripts
COPY docker/emotion-features/scripts/ /app/scripts/

# Set working directory
WORKDIR /app
