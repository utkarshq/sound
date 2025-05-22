"""Unit tests for audio processing module."""
import pytest
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.audio_processing import AudioProcessor, AudioProcessorError
from src.config.config_manager import AudioConfig
from tests.base_test import BaseTest

class TestAudioProcessor(BaseTest):
    """Test cases for AudioProcessor."""

    class TestProcessor(AudioProcessor):
        """Test implementation of AudioProcessor with required process method."""        def process(self, input_data: Any) -> Dict[str, Any]:
            """Implementation of abstract process method for testing."""
            if isinstance(input_data, str):
                result = self.load_audio(input_data)
                waveform = result[0]
                original_sr = result[1].get('original_sr', self.config.target_sr)
            else:
                waveform = input_data
                original_sr = self.config.target_sr
                
            # Basic processing pipeline
            if original_sr != self.config.target_sr:
                waveform = self.resample(waveform, original_sr)
            
            if self.config.normalize:
                waveform = self.normalize_loudness(waveform)
                
            if self.config.remove_silence:
                waveform = self.remove_silence(waveform)
                
            spec = self.compute_spectrogram(waveform)
            
            return {
                "waveform": waveform,
                "spectrogram": spec,
                "sample_rate": self.config.target_sr,
                "metadata": {"processed": True}
            }

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

    @pytest.fixture
    def processor(self, config_manager):
        """Create TestProcessor instance for testing."""
        return self.TestProcessor(config_manager=config_manager)

    def test_audio_config_validation(self):
        """Test AudioConfig validation."""
        # Test negative sample rate
        with pytest.raises(ValueError):
            AudioConfig(target_sr=-16000)
        
        # Test positive loudness (should be negative LUFS)
        with pytest.raises(ValueError):
            AudioConfig(target_loudness=0)
        
        # Test negative segment duration
        with pytest.raises(ValueError):
            AudioConfig(min_segment_dur=-0.1)

    def test_processor_initialization(self, processor):
        """Test AudioProcessor initialization and properties."""
        # Basic initialization checks
        assert processor.config.target_sr == 16000
        assert processor.config.target_loudness == -23.0
        assert processor.config.n_mels == 80
        
        # Check device setup
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert processor.device == expected_device
        
        # Check cache initialization
        assert isinstance(processor._resampler_cache, dict)

    def test_audio_loading(self, processor, tmp_path):
        """Test audio loading with validation."""
        # Create test file
        test_path = self.create_test_audio_file(tmp_path / "test.wav", sample_rate=44100)
        
        # Test successful loading
        result = processor.load_audio(test_path)
        waveform = result[0]
        metadata = result[1]
        sr = metadata.get('original_sr', self.config_manager.get_pipeline_config().audio.target_sr)
        self.assert_audio_valid(waveform, check_mono=True)
        assert sr == 44100
        
        # Test loading non-existent file
        with pytest.raises(AudioProcessorError):
            processor.load_audio(tmp_path / "nonexistent.wav")

    def test_audio_preprocessing(self, processor, tmp_path):
        """Test audio preprocessing pipeline."""
        # Create test audio
        test_path = self.create_test_audio_file(
            tmp_path / "test.wav",
            sample_rate=44100,
            duration=2.0
        )
        
        # Load and process
        result = processor.load_audio(test_path)
        waveform = result[0]
        metadata = result[1]
        sr = metadata.get('original_sr', self.config_manager.get_pipeline_config().audio.target_sr)
        
        # Test resampling
        resampled = processor.resample(waveform, sr)
        expected_length = int(waveform.shape[1] * processor.config.target_sr / sr)
        assert resampled.shape[1] == expected_length
        
        # Test normalization
        normalized = processor.normalize_loudness(resampled)
        assert torch.abs(normalized).max() <= 1.0
        
        # Test silence removal
        no_silence = processor.remove_silence(normalized)
        assert no_silence.shape[1] < normalized.shape[1]
        
        # Test spectrogram computation
        spec = processor.compute_spectrogram(no_silence)
        assert spec.dim() == 3
        assert spec.shape[1] == processor.config.n_mels

    def test_batch_processing(self, processor, tmp_path):
        """Test batch processing capabilities."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            path = tmp_path / f"test_{i}.wav"
            self.create_test_audio_file(
                path,
                sample_rate=44100,
                duration=1.0 + i * 0.5
            )
            test_files.append(str(path))
        
        # Process batch
        results = processor.process_batch(test_files)
        
        # Validate results
        assert len(results) == len(test_files)
        for result in results:
            assert "waveform" in result
            assert "spectrogram" in result
            assert "metadata" in result
            self.assert_audio_valid(result["waveform"])
            assert result["spectrogram"].shape[1] == processor.config.n_mels
