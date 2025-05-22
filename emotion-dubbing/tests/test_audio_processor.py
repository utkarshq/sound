"""Tests for the AudioProcessor class.

Tests the audio processing functionality including loading, normalization,
silence removal, and spectrogram computation.
"""

import pytest
import torch
import torchaudio
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
from typing import Dict, Any

from src.audio_processing import AudioProcessor, AudioProcessorError
from src.config.config_manager import ConfigManager
from tests.base_test import BaseTest

class TestAudioProcessor(BaseTest):
    class TestProcessor(AudioProcessor):
        """Test implementation of AudioProcessor with required process method."""
        def process(self, input_data: Any) -> Dict[str, Any]:
            """Implementation of abstract process method for testing."""
            if isinstance(input_data, str):
                waveform, sr = self.load_audio(input_data)
            else:
                waveform = input_data
                sr = self.config.target_sr
                
            # Run through processing pipeline
            waveform = self.process_audio(waveform, sr)
            spec = self.compute_spectrogram(waveform)
            
            return {
                "waveform": waveform,
                "spectrogram": spec,
                "metadata": {
                    "sample_rate": self.config.target_sr,
                    "duration": waveform.shape[1] / self.config.target_sr
                }
            }
            
        def process_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
            """Helper method to process audio through the pipeline."""
            if sr != self.config.target_sr:
                waveform = self.resample(waveform, sr)
            if self.config.normalize:
                waveform = self.normalize_loudness(waveform)
            if self.config.remove_silence:
                waveform = self.remove_silence(waveform)
            return waveform

        def resample(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
            """Resample audio to target sample rate."""
            if original_sr == self.config.target_sr:
                return waveform
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr,
                new_freq=self.config.target_sr
            ).to(self.device)
            return resampler(waveform)

        def process_batch(self, inputs: list) -> list:
            """Process a batch of inputs."""
            results = []
            for input_data in inputs:
                results.append(self.process(input_data))
            return results

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.config_manager = ConfigManager(config_dir=str(Path(__file__).parent / "data"))
        self.processor = self.TestProcessor(config_manager=self.config_manager)
        
        # Create test audio files with different characteristics
        self.test_files = {}
        
        # Normal audio file
        self.test_files['normal'] = self.create_test_audio(
            duration=2.0,
            frequencies=[440],
            sample_rate=44100
        )
        
        # Multi-frequency audio
        self.test_files['complex'] = self.create_test_audio(
            duration=2.0,
            frequencies=[440, 880, 1760],
            sample_rate=48000
        )
        
        # Very short audio
        self.test_files['short'] = self.create_test_audio(
            duration=0.05,
            frequencies=[440],
            sample_rate=16000
        )
        
        yield
        
        # Cleanup
        for path in self.test_files.values():
            Path(path).unlink(missing_ok=True)
        self.processor._clear_cache()

    def create_test_audio(self, duration, frequencies, sample_rate):
        """Create a test audio file with specified properties."""
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio = np.zeros_like(t)
        
        for freq in frequencies:
            audio += np.sin(2 * np.pi * freq * t)
        
        # Add silence region
        silence_start = int(duration * sample_rate * 0.4)
        silence_end = int(duration * sample_rate * 0.6)
        audio[silence_start:silence_end] = 0
        
        # Add noise
        audio += np.random.normal(0, 0.01, len(t))
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix='.wav')
        sf.write(temp_path, audio, sample_rate)
        return temp_path

    def test_load_and_resample(self):
        """Test audio loading and resampling."""
        for name, path in self.test_files.items():
            waveform, sr = self.processor.load_audio(path)
            assert isinstance(waveform, torch.Tensor)
            assert waveform.dim() == 2
            
            resampled = self.processor.resample(waveform, sr)
            assert resampled.shape[1] == int(waveform.shape[1] * self.processor.config.target_sr / sr)

    def test_loudness_normalization(self):
        """Test loudness normalization."""
        for name, path in self.test_files.items():
            waveform, sr = self.processor.load_audio(path)
            resampled = self.processor.resample(waveform, sr)
            normalized = self.processor.normalize_loudness(resampled)
            
            # Check that output is normalized
            assert torch.abs(normalized).max() <= 1.0
            
            # Verify approximate target loudness
            loudness = self.processor.loudness_meter.integrated_loudness(normalized.cpu().numpy())
            assert abs(loudness - self.processor.config.target_loudness) < 1.0

    def test_silence_removal(self):
        """Test silence removal."""
        for name, path in self.test_files.items():
            if name == 'short':
                continue  # Skip short file
                
            waveform, sr = self.processor.load_audio(path)
            resampled = self.processor.resample(waveform, sr)
            normalized = self.processor.normalize_loudness(resampled)
            no_silence = self.processor.remove_silence(normalized)
            
            # Output should be shorter
            assert no_silence.shape[1] < normalized.shape[1]
            
            # Check that remaining audio has sufficient energy
            rms = torch.sqrt(torch.mean(no_silence ** 2))
            db = 20 * torch.log10(rms)
            assert db > self.processor.config.silence_threshold_db

    def test_spectrogram_computation(self):
        """Test mel spectrogram computation."""
        for name, path in self.test_files.items():
            waveform, sr = self.processor.load_audio(path)
            resampled = self.processor.resample(waveform, sr)
            
            spec = self.processor.compute_spectrogram(resampled)
            assert spec.dim() == 3
            assert spec.shape[1] == self.processor.config.n_mels
            assert not torch.isnan(spec).any()
            assert not torch.isinf(spec).any()

    def test_error_handling(self):
        """Test error handling for various scenarios."""
        # Test nonexistent file
        with pytest.raises(AudioProcessorError):
            self.processor.load_audio("nonexistent.wav")
        
        # Test invalid audio data
        with pytest.raises(AudioProcessorError):
            self.processor.normalize_loudness(torch.zeros(1, 0))
            
        # Test incompatible sample rate
        with pytest.raises(ValueError):
            self.processor.resample(torch.randn(1, 1000), -16000)

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        file_paths = list(self.test_files.values())
        results = self.processor.process_batch(file_paths)
        
        assert len(results) == len(file_paths)
        for result in results:
            assert all(key in result for key in ['waveform', 'spectrogram', 'metadata'])
            assert isinstance(result['waveform'], torch.Tensor)
            assert isinstance(result['spectrogram'], torch.Tensor)
            assert isinstance(result['metadata'], dict)
