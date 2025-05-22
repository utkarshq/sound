import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
import numpy as np
import json
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProsodyAnalyzer:
    def __init__(self, config_path="/app/config/pipeline.yaml"):
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["pipeline"]["emotion_analysis"]
        
        # Initialize feature extractor
        model_path = Path("/models/prosody")
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_path))
        else:
            logger.info("Loading model from HuggingFace")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            )
    
    def analyze_prosody(self, audio_path: str):
        """Analyze prosody features from audio file using configured parameters"""
        logger.info(f"Analyzing prosody for {audio_path}")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Calculate windows using configuration
        window_size = self.config["window_size"]
        overlap = self.config["overlap"]
        samples_per_window = int(window_size * sample_rate)
        hop_length = int(samples_per_window * (1 - overlap))
        
        # Create overlapping windows
        windows = []
        for start in range(0, waveform.size(1), hop_length):
            end = start + samples_per_window
            if end <= waveform.size(1):
                windows.append(waveform[0, start:end])
        
        results = {
            "pitch_contour": [],
            "energy_contour": [],
            "emotion_frames": []
        }
        
        for window in windows:
            if len(window) < samples_per_window:
                continue
                
            # Extract features
            features = self.feature_extractor(window, sampling_rate=sample_rate)
            
            # Calculate pitch and energy
            pitch = self._calculate_pitch(window, sample_rate)
            energy = torch.norm(window).item()
            
            results["pitch_contour"].append(pitch)
            results["energy_contour"].append(energy)
        
        return results

    def _calculate_pitch(self, waveform: torch.Tensor, sample_rate: int):
        """Calculate pitch using autocorrelation"""
        # Convert to numpy for easier processing
        signal = waveform.numpy()
        
        # Parameters for pitch detection
        min_freq = 50  # Hz
        max_freq = 500  # Hz
        
        # Calculate lags to check based on frequency range
        min_lag = int(sample_rate / max_freq)
        max_lag = int(sample_rate / min_freq)
        
        # Normalize signal
        signal = signal - np.mean(signal)
        signal = signal / (np.std(signal) + 1e-6)
        
        # Calculate autocorrelation
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(min_lag, min(max_lag, len(corr)-1)):
            if corr[i] > corr[i-1] and corr[i] > corr[i+1]:
                peaks.append((i, corr[i]))
        
        # Find highest peak in valid range
        if peaks:
            lag = max(peaks, key=lambda x: x[1])[0]
            pitch = sample_rate / lag
            return float(pitch)
        else:
            return 0.0  # No clear pitch detected

if __name__ == "__main__":
    analyzer = ProsodyAnalyzer()
    results = analyzer.analyze_prosody("/output/audio.wav")
    
    with open("/output/prosody_analysis.json", "w") as f:
        json.dump(results, f, indent=2)