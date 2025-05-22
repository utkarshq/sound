import logging
from pathlib import Path
import yaml

def setup_logging(config_path="/app/config/pipeline.yaml"):
    """Setup logging based on configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logging_config = config.get("logging", {})
    log_level = getattr(logging, logging_config.get("level", "INFO"))
    log_file = logging_config.get("file")
    
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

class PipelineError(Exception):
    """Custom exception for pipeline errors"""
    pass

def validate_config(config_path="/app/config/pipeline.yaml"):
    """Validate pipeline configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    required_sections = ["pipeline", "logging"]
    for section in required_sections:
        if section not in config:
            raise PipelineError(f"Missing required section: {section}")
    
    pipeline_config = config["pipeline"]
    required_pipeline_sections = [
        "audio_extraction",
        "transcription",
        "diarization",
        "emotion_analysis",
        "output"
    ]
    
    for section in required_pipeline_sections:
        if section not in pipeline_config:
            raise PipelineError(f"Missing required pipeline section: {section}")
    
    return config

def ensure_directories():
    """Ensure required directories exist"""
    dirs = ["input", "output", "models"]
    for dir_name in dirs:
        path = Path(dir_name)
        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"Created directory: {path}")
        elif not path.is_dir():
            raise PipelineError(f"Path exists but is not a directory: {path}")
    return True
        Path(dir_name).mkdir(parents=True, exist_ok=True)
