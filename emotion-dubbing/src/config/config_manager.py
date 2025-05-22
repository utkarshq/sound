"""Configuration manager for emotion dubbing pipeline."""
import os
from pathlib import Path
import yaml
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    source: str
    type: str

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    target_sr: int = 16000
    target_loudness: float = -23.0
    silence_threshold_db: float = -40.0
    min_segment_dur: float = 0.1
    use_gpu: bool = True
    # FFT parameters
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 80
    # Sample conversion
    input_sr: int = 44100
    channels: int = 1
    format: str = "wav"
    # Processing
    normalize: bool = True
    remove_silence: bool = True
    window_size: float = 0.5
    overlap: float = 0.25

    def __post_init__(self):
        """Validate configuration values."""
        if self.target_sr <= 0:
            raise ValueError(f"Target sample rate must be positive, got {self.target_sr}")
        if self.target_loudness >= 0:
            raise ValueError(f"Target loudness must be negative LUFS, got {self.target_loudness}")
        if self.min_segment_dur <= 0:
            raise ValueError(f"Minimum segment duration must be positive, got {self.min_segment_dur}")
        if self.n_fft <= 0 or not (self.n_fft & (self.n_fft - 1) == 0):
            raise ValueError(f"FFT size must be positive power of 2, got {self.n_fft}")
        if self.hop_length <= 0:
            raise ValueError(f"Hop length must be positive, got {self.hop_length}")
        if self.n_mels <= 0:
            raise ValueError(f"Number of mel bands must be positive, got {self.n_mels}")
        if self.channels <= 0:
            raise ValueError(f"Number of channels must be positive, got {self.channels}")
        if self.window_size <= 0:
            raise ValueError(f"Window size must be positive, got {self.window_size}")
        if not 0 <= self.overlap < 1:
            raise ValueError(f"Overlap must be in [0, 1), got {self.overlap}")
        
        # Convert string values from YAML
        if isinstance(self.target_sr, str):
            self.target_sr = int(self.target_sr)
        if isinstance(self.input_sr, str):
            self.input_sr = int(self.input_sr)
        if isinstance(self.n_fft, str):
            self.n_fft = int(self.n_fft)
        if isinstance(self.hop_length, str):
            self.hop_length = int(self.hop_length)
        if isinstance(self.n_mels, str):
            self.n_mels = int(self.n_mels)
        if isinstance(self.channels, str):
            self.channels = int(self.channels)

@dataclass
class EmotionConfig:
    """Emotion analysis configuration."""
    window_size: float = 0.5
    overlap: float = 0.25
    feature_extractors: List[str] = None

    def __post_init__(self):
        if self.feature_extractors is None:
            self.feature_extractors = ['wav2vec2', 'opensmile']
        if not 0 <= self.overlap < 1:
            raise ValueError(f"Overlap must be in [0, 1), got {self.overlap}")

@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    audio: AudioConfig
    emotion: EmotionConfig
    models_path: str = '/models'
    output_path: str = '/output'
    cache_models: bool = True
    verify_downloads: bool = True

    @property
    def emotion_analysis(self):
        """Alias for emotion config for backward compatibility."""
        return self.emotion

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory. If None, uses default paths.
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / 'config'
        self.models_config: Dict[str, ModelConfig] = {}
        self.pipeline_config: Optional[PipelineConfig] = None
        self._load_configs()

    def _load_configs(self):
        """Load all configuration files."""
        try:
            # Load models config
            models_path = self.config_dir / 'models.yaml'
            if not models_path.exists():
                raise FileNotFoundError(f"Models configuration not found at {models_path}")
                
            with open(models_path) as f:
                models_data = yaml.safe_load(f)
                
            for model_id, model_data in models_data.get('models', {}).items():
                self.models_config[model_id] = ModelConfig(**model_data)

            # Load pipeline config
            pipeline_path = self.config_dir / 'pipeline.yaml'
            if not pipeline_path.exists():
                raise FileNotFoundError(f"Pipeline configuration not found at {pipeline_path}")
                
            with open(pipeline_path) as f:
                pipeline_data = yaml.safe_load(f)
                
            self.pipeline_config = PipelineConfig(
                audio=AudioConfig(**pipeline_data['pipeline'].get('audio_extraction', {})),
                emotion=EmotionConfig(**pipeline_data['pipeline'].get('emotion_analysis', {})),
                models_path=pipeline_data.get('model_path', '/models'),
                output_path=pipeline_data.get('output_path', '/output'),
                cache_models=pipeline_data.get('cache_models', True),
                verify_downloads=pipeline_data.get('verify_downloads', True)
            )

            logger.info("Configurations loaded successfully")

        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise

    def validate_config(self) -> bool:
        """Validate loaded configurations.
        
        Returns:
            bool: True if configuration is valid.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate models config
        required_models = {'whisper', 'diarization', 'emotion'}
        missing_models = required_models - set(self.models_config.keys())
        if missing_models:
            raise ValueError(f"Missing required model configurations: {missing_models}")

        # Validate model sources
        for model_id, model_config in self.models_config.items():
            if not model_config.source or not model_config.type:
                raise ValueError(f"Invalid configuration for model {model_id}: missing source or type")

        # Validate pipeline config
        if not self.pipeline_config:
            raise ValueError("Pipeline configuration not loaded")

        # Validate paths
        models_path = Path(self.pipeline_config.models_path)
        output_path = Path(self.pipeline_config.output_path)
        
        if not models_path.exists():
            models_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created models directory: {models_path}")
            
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_path}")

        return True

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Get configuration for a specific model.
        
        Args:
            model_id: ID of the model.
            
        Returns:
            ModelConfig: Configuration for the model.
            
        Raises:
            KeyError: If model_id is not found.
        """
        if model_id not in self.models_config:
            raise KeyError(f"Model configuration not found: {model_id}")
        return self.models_config[model_id]

    def get_pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration.
        
        Returns:
            PipelineConfig: Pipeline configuration.
            
        Raises:
            ValueError: If pipeline configuration is not loaded.
        """
        if not self.pipeline_config:
            raise ValueError("Pipeline configuration not loaded")
        return self.pipeline_config
