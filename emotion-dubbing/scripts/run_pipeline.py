#!/usr/bin/env python3
import os
import sys
import logging
import subprocess
import yaml
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class Pipeline:
    def __init__(self):
        self.logger = setup_logging()
        self.config = self.load_config()
    
    def load_config(self):
        config_path = Path(__file__).parent.parent / "config" / "pipeline.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def run_command(self, command, description):
        self.logger.info(f"=== {description} ===")
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            self.logger.error(f"Failed: {description}")
            sys.exit(1)
    
    def run(self):
        # Check if models are ready
        self.run_command(
            "docker compose run --rm model-hub python /app/scripts/model_manager.py",
            "Checking Models"
        )

        # Extract audio
        self.run_command(
            "docker compose run --rm ffmpeg -i /input/test.mp4 -vn -acodec pcm_s16le -ar 16000 /output/audio.wav",
            "Extracting Audio"
        )        # Run parallel processing
        self.run_command(
            "docker compose run --rm whisper --model large-v3 --language ja --output_dir /output --output_format json /output/audio.wav",
            "Running Transcription"
        )
        
        self.run_command(
            "docker compose run --rm nemo-diarize python /app/diarize.py /output/audio.wav /output/diarization.json",
            "Running Speaker Diarization"
        )

        # Process features
        commands = [
            ("docker compose run --rm emotion-features python /app/scripts/align_speakers.py", "Aligning Speakers"),
            ("docker compose run --rm emotion-features python /app/scripts/extract_wav2vec2.py --input /output/audio.wav --segments /output/aligned.json", "Extracting Wav2Vec2 Features"),
            ("docker compose run --rm emotion-features python /app/scripts/analyze_prosody.py", "Analyzing Prosody")
        ]

        for cmd, desc in commands:
            self.run_command(cmd, desc)

        # Process final results
        self.run_command(
            "docker compose run --rm emotion-features python /app/scripts/process_opensmile.py",
            "Processing Final Results"
        )

        self.logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
