"""Base processor class for emotion dubbing pipeline."""
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import torch

from src.config.config_manager import ConfigManager

class ProcessorError(Exception):
    """Base class for processor errors."""
    pass

class ModelError(ProcessorError):
    """Error related to model operations."""
    pass

class ConfigError(ProcessorError):
    """Error related to configuration."""
    pass

class BaseProcessor(ABC):
    """Abstract base class for all processors in the pipeline."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize base processor.
        
        Args:
            config_manager: Configuration manager instance. If None, creates new one.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager or ConfigManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up GPU optimizations if available
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
        
        self.logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")

    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data.
        
        Args:
            input_data: Input data to process.
            
        Returns:
            Dict[str, Any]: Processing results.
            
        Raises:
            ProcessorError: If processing fails.
        """
        raise NotImplementedError

    def _validate_input(self, input_data: Any) -> bool:
        """Validate input data.
        
        Args:
            input_data: Input data to validate.
            
        Returns:
            bool: True if input is valid.
            
        Raises:
            ValueError: If input is invalid.
        """
        if input_data is None:
            raise ValueError("Input data cannot be None")
        return True

    def _ensure_paths(self, paths: Dict[str, str]):
        """Ensure required paths exist.
        
        Args:
            paths: Dictionary of path names and their values.
        """
        for name, path in paths.items():
            path = Path(path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created {name} directory: {path}")

    def to_device(self, data: torch.Tensor) -> torch.Tensor:
        """Move tensor to appropriate device.
        
        Args:
            data: Input tensor.
            
        Returns:
            torch.Tensor: Tensor on appropriate device.
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return data

    def cleanup(self):
        """Clean up resources."""
        # Base implementation - override if needed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def __enter__(self):
        """Context manager enter."""
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        self.cleanup()
