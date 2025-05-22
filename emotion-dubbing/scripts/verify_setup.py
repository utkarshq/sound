#!/usr/bin/env python3
import os
import sys
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_structure():
    """Verify project structure and configurations"""
    required_dirs = ['input', 'output', 'config', 'docker', 'scripts']
    required_files = [
        'config/models.yaml',
        'config/pipeline.yaml',
        'docker-compose.yml',
        'docker/whisper-jp/Dockerfile',
        'docker/nemo-diarize/Dockerfile',
        'docker/emotion-features/Dockerfile',
        'docker/model-hub/Dockerfile'
    ]
    
    # Check directories
    for dir_name in required_dirs:
        if not Path(dir_name).is_dir():
            logger.error(f"Missing directory: {dir_name}")
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).is_file():
            logger.error(f"Missing file: {file_path}")
            return False

    return True

def verify_configs():
    """Verify configuration files"""
    try:
        # Check models config
        with open('config/models.yaml') as f:
            models_config = yaml.safe_load(f)
            required_models = ['whisper', 'diarization', 'emotion', 'prosody']
            for model in required_models:
                if model not in models_config.get('models', {}):
                    logger.error(f"Missing model configuration for: {model}")
                    return False

        # Check pipeline config
        with open('config/pipeline.yaml') as f:
            pipeline_config = yaml.safe_load(f)
            required_sections = ['audio_extraction', 'transcription', 'diarization', 'emotion_analysis']
            for section in required_sections:
                if section not in pipeline_config.get('pipeline', {}):
                    logger.error(f"Missing pipeline configuration for: {section}")
                    return False

        return True

    except Exception as e:
        logger.error(f"Error verifying configurations: {str(e)}")
        return False

def verify_requirements():
    """Verify requirements files"""
    docker_dirs = ['emotion-features', 'model-hub']
    for dir_name in docker_dirs:
        req_file = f'docker/{dir_name}/requirements.txt'
        if not Path(req_file).is_file():
            logger.error(f"Missing requirements file: {req_file}")
            return False
    return True

def main():
    logger.info("Verifying project setup...")
    
    if not verify_structure():
        logger.error("Project structure verification failed")
        return 1
    
    if not verify_configs():
        logger.error("Configuration verification failed")
        return 1
    
    if not verify_requirements():
        logger.error("Requirements verification failed")
        return 1
    
    logger.info("All verifications passed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
