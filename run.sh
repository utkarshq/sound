#!/bin/bash

# Create main directories
mkdir -p emotion-dubbing/input
mkdir -p emotion-dubbing/output
mkdir -p emotion-dubbing/scripts
mkdir -p emotion-dubbing/docker/whisper-jp
mkdir -p emotion-dubbing/docker/nemo-diarize
mkdir -p emotion-dubbing/docker/emotion-features

# Create placeholder files
touch emotion-dubbing/input/test.mp4

touch emotion-dubbing/scripts/1_transcribe_diarize_emotion.sh
touch emotion-dubbing/scripts/align_speakers.py
touch emotion-dubbing/scripts/extract_wav2vec2.py
touch emotion-dubbing/scripts/process_opensmile.py

touch emotion-dubbing/docker/whisper-jp/Dockerfile

touch emotion-dubbing/docker/nemo-diarize/Dockerfile
touch emotion-dubbing/docker/nemo-diarize/diarize.py

touch emotion-dubbing/docker/emotion-features/Dockerfile
touch emotion-dubbing/docker/emotion-features/requirements.txt

touch emotion-dubbing/docker-compose.yml
touch emotion-dubbing/README.md

echo "Directory structure and placeholder files created."
