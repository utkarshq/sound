#!/bin/bash
set -e

# Change to script directory
cd "$(dirname "$0")"
cd ..

# Ensure required directories exist
mkdir -p input output

# Function to check docker compose version
check_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    else
        echo "docker compose"
    fi
}

# Get docker compose command
DOCKER_COMPOSE=$(check_docker_compose)

# Function to check NVIDIA GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Install required Python packages if not present
python3 -m pip install --user pyyaml

# Check if input file exists
if [ ! -f "input/test.mp4" ]; then
    echo "Error: No input file found at input/test.mp4"
    exit 1
fi

# Run the Python pipeline script
python3 scripts/run_pipeline.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline failed. Check logs for details."
    exit 1
fi
