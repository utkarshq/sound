"""Base test class for emotion dubbing pipeline tests."""
import pytest
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from src.config.config_manager import ConfigManager, AudioConfig
from src.processors.base_processor import BaseProcessor

class BaseTest:
    """Base class for all test cases."""
    
    @pytest.fixture
    def config_manager(self):
        """Create test configuration manager."""
        config_dir = Path(__file__).parent / "data"
        return ConfigManager(config_dir=str(config_dir))

    @pytest.fixture
    def audio_config(self):
        """Create test audio configuration."""
        return AudioConfig(
            target_sr=16000,
            target_loudness=-23.0,
            silence_threshold_db=-40.0,
            min_segment_dur=0.1,
            use_gpu=True,
            n_fft=2048,
            hop_length=512,
            n_mels=80
        )

    @pytest.fixture
    def test_audio(self) -> torch.Tensor:
        """Create synthetic test audio."""
        duration = 2  # seconds
        sample_rate = 16000
        t = torch.linspace(0, duration, int(duration * sample_rate))
        # Generate complex test signal with harmonics
        freqs = [440, 880]  # Fundamental and first harmonic
        audio = torch.zeros_like(t)
        for i, f in enumerate(freqs):
            audio += (0.5 ** (i+1)) * torch.sin(2 * np.pi * f * t)
        # Add silence in the middle
        silence_start = int(duration * sample_rate * 0.4)
        silence_end = int(duration * sample_rate * 0.6)
        audio[silence_start:silence_end] = 0
        return audio.unsqueeze(0)  # Add channel dimension

    def create_test_audio_file(self, path: Path, sample_rate: int = 44100, duration: float = 2.0) -> Path:
        """Create a test audio file with specific characteristics.
        
        Args:
            path: Output path
            sample_rate: Sample rate in Hz
            duration: Duration in seconds
        
        Returns:
            Path: Path to created audio file
        """
        t = torch.linspace(0, duration, int(duration * sample_rate))
        freqs = [440, 880]
        audio = torch.zeros_like(t)
        for i, f in enumerate(freqs):
            audio += (0.5 ** (i+1)) * torch.sin(2 * np.pi * f * t)
        
        # Add silence segment
        silence_start = int(duration * sample_rate * 0.4)
        silence_end = int(duration * sample_rate * 0.6)
        audio[silence_start:silence_end] = 0
        
        # Add some noise
        noise = torch.randn_like(audio) * 0.01
        audio += noise
        
        # Normalize
        audio = audio / torch.max(torch.abs(audio))
        
        # Save
        torchaudio.save(str(path), audio.unsqueeze(0), sample_rate)
        return path

    @staticmethod
    def assert_audio_valid(audio: torch.Tensor, check_mono: bool = True, expected_sr: Optional[int] = None):
        """Assert that audio tensor is valid.
        
        Args:
            audio: Audio tensor to validate
            check_mono: Whether to check if audio is mono
            expected_sr: Expected sample rate to validate against
        """
        assert isinstance(audio, torch.Tensor)
        assert not torch.isnan(audio).any()
        assert not torch.isinf(audio).any()
        assert audio.dim() >= 2  # At least [channels, samples]
        if check_mono:
            assert audio.size(0) == 1  # Mono audio

    @staticmethod
    def generate_test_audio(duration: float = 1.0, sample_rate: int = 16000, 
                           frequencies: Optional[Tuple[float, ...]] = None) -> torch.Tensor:
        """Generate test audio with multiple frequency components.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            frequencies: Tuple of frequencies in Hz (default: [440, 880])
            
        Returns:
            torch.Tensor: Generated audio signal
        """
        if frequencies is None:
            frequencies = (440.0, 880.0)
            
        t = torch.linspace(0, duration, int(duration * sample_rate))
        signal = torch.zeros_like(t)
        
        for freq in frequencies:
            signal += torch.sin(2 * np.pi * freq * t)
            
        return signal.unsqueeze(0) / len(frequencies)  # Normalize and add channel dim

    @staticmethod
    def assert_spectrogram_valid(spec: torch.Tensor):
        """Assert that spectrogram tensor is valid.
        
        Args:
            spec: Spectrogram tensor to validate
        """
        assert isinstance(spec, torch.Tensor)
        assert not torch.isnan(spec).any()
        assert not torch.isinf(spec).any()
        assert spec.dim() == 3  # (channels, freq, time)

    @staticmethod
    def assert_metadata_valid(metadata: dict, required_keys: Optional[list] = None):
        """Assert that metadata dictionary is valid.
        
        Args:
            metadata: Metadata dictionary to validate
            required_keys: List of required keys
        """
        assert isinstance(metadata, dict)
        if required_keys:
            for key in required_keys:
                assert key in metadata
