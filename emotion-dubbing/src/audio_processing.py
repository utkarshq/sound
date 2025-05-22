import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pyloudnorm
import torch.nn.functional as F

import logging
from src.processors.base_processor import BaseProcessor, ProcessorError
from src.config.config_manager import ConfigManager, AudioConfig

logger = logging.getLogger(__name__)

class AudioProcessorError(ProcessorError):
    """Audio processor specific errors."""
    pass

class AudioProcessor(BaseProcessor):
    """Audio processor for handling audio processing tasks with GPU acceleration.
    
    Handles audio loading, resampling, normalization, silence removal, spectrogram
    computation, and prosody analysis with proper error handling and GPU memory management.
    """
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize audio processor.
        
        Args:
            config_manager: Configuration manager instance. If None, creates new one.
        """
        super().__init__(config_manager)
        
        # Get configurations
        pipeline_config = self.config_manager.get_pipeline_config()
        self.config = pipeline_config.audio
        self.emotion_config = pipeline_config.emotion_analysis
        
        # Cache transforms
        self._resampler_cache = {}
        self._mel_spec_transform = None
        
        # Initialize audio quality meters
        self.loudness_meter = pyloudnorm.Meter(self.config.target_sr)
        
        # Set up prosody analysis
        self._setup_prosody_analysis()
        
        # Track memory usage
        self._memory_counter = 0
        self._max_memory_gb = 4.0  # Default 4GB limit
        
        # Ensure output directory exists
        self._ensure_paths({'output': pipeline_config.output_path})

    def _setup_prosody_analysis(self):
        """Initialize prosody analysis components."""
        try:
            # Initialize OpenSmile
            import opensmile
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            
            # Initialize wav2vec2 feature extractor if available
            try:
                from transformers import Wav2Vec2FeatureExtractor
                model_path = Path("/models/prosody")
                if model_path.exists():
                    logger.info(f"Loading wav2vec2 model from {model_path}")
                    self.wav2vec2_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_path))
                else:
                    logger.info("Loading wav2vec2 model from HuggingFace")
                    self.wav2vec2_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
                    )
            except ImportError:
                logger.warning("Wav2Vec2 not available. Some features will be disabled.")
                self.wav2vec2_extractor = None
                
        except ImportError:
            logger.warning("OpenSmile not available. Some features will be disabled.")
            self.smile = None
            self.wav2vec2_extractor = None

    def __enter__(self):
        super().__enter__()
        torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._clear_cache()
        super().__exit__(exc_type, exc_val, exc_tb)

    def _clear_cache(self):
        """Clear cached transforms and GPU memory"""
        self._resampler_cache.clear()
        self._mel_spec_transform = None
        torch.cuda.empty_cache()
        self._memory_counter = 0

    def _check_memory(self, tensor_size: int):
        """Check if processing will exceed memory limits"""
        self._memory_counter += tensor_size / (1024 ** 3)  # Convert to GB
        if self._memory_counter > self._max_memory_gb:
            self._clear_cache()
            torch.cuda.empty_cache()
            self._memory_counter = tensor_size / (1024 ** 3)

    def _validate_audio(self, waveform: torch.Tensor, sr: int) -> None:
        """Validate audio tensor and sampling rate"""
        if not isinstance(waveform, torch.Tensor):
            raise AudioProcessorError("Input waveform must be a torch.Tensor")
        
        if waveform.dim() not in [1, 2]:
            raise AudioProcessorError("Waveform must be 1D or 2D tensor")
            
        if sr <= 0 or sr > 192000:
            raise AudioProcessorError(f"Invalid sampling rate: {sr}")

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        """Get cached resampler or create new one with error handling"""
        try:
            key = (orig_sr, self.config.target_sr)
            if key not in self._resampler_cache:
                self._resampler_cache[key] = torchaudio.transforms.Resample(
                    orig_sr, self.config.target_sr
                ).to(self.device)
            return self._resampler_cache[key]
        except Exception as e:
            raise AudioProcessorError(f"Failed to create resampler: {str(e)}")

    def load_audio(self, input_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Load audio file and return tensor with metadata.
        
        Args:
            input_path: Path to audio file
            
        Returns:
            Tuple of (waveform tensor, metadata dict)
            
        Raises:
            AudioProcessorError: If file cannot be loaded or is invalid
        """
        try:
            if not Path(input_path).exists():
                raise AudioProcessorError(f"Audio file not found: {input_path}")
                
            waveform, sr = torchaudio.load(input_path)
            self._validate_audio(waveform, sr)
            
            waveform = waveform.to(self.device)
            self._check_memory(waveform.element_size() * waveform.nelement())
            
            metadata = {
                'original_sr': sr,
                'channels': waveform.shape[0],
                'duration': waveform.shape[1] / sr,
                'file_path': input_path
            }
            
            if sr != self.config.target_sr:
                waveform = self._get_resampler(sr)(waveform)
                
            return waveform, metadata
            
        except Exception as e:
            raise AudioProcessorError(f"Error loading audio file {input_path}: {str(e)}")

    def convert_to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel audio to mono using weighted average"""
        if waveform.shape[0] > 1:
            # Use learned channel weights for better mono conversion
            weights = torch.tensor([0.7, 0.3] if waveform.shape[0] == 2 else
                                 [1.0/waveform.shape[0]] * waveform.shape[0],
                                 device=self.device)
            return (waveform * weights.view(-1, 1)).sum(dim=0, keepdim=True)
        return waveform

    @torch.no_grad()
    def normalize_loudness(self, waveform: torch.Tensor, fast_mode: bool = False) -> torch.Tensor:
        """Normalize audio loudness with peak limiting and optional fast mode.
        
        Args:
            waveform: Input audio tensor
            fast_mode: If True, uses simplified normalization for speed
            
        Returns:
            Normalized audio tensor
        """
        try:
            if fast_mode:
                # Simple peak normalization
                peak = torch.max(torch.abs(waveform))
                return waveform * (0.95 / peak) if peak > 0 else waveform
            
            # Full loudness normalization
            audio_np = waveform.cpu().numpy().T
            current_loudness = self.loudness_meter.integrated_loudness(audio_np)
            
            # Calculate gain with peak limiting
            gain_db = self.config.target_loudness - current_loudness
            gain_linear = 10 ** (gain_db / 20)
            
            # Apply soft knee limiter with oversampling
            waveform = waveform * gain_linear
            peak_threshold = 0.99
            
            if torch.max(torch.abs(waveform)) > peak_threshold:
                # Oversample for better limiting
                waveform = F.interpolate(
                    waveform.unsqueeze(1), 
                    scale_factor=2, 
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                waveform = torch.tanh(waveform)
                waveform = F.interpolate(
                    waveform.unsqueeze(1),
                    size=waveform.shape[-1] // 2,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
            
            return waveform
            
        except Exception as e:
            raise AudioProcessorError(f"Error in loudness normalization: {str(e)}")

    def remove_silence(
        self,
        waveform: torch.Tensor,
        pad_ms: int = 50
    ) -> torch.Tensor:
        """Remove silence with adaptive thresholding and padding"""
        # Convert to numpy for librosa processing
        audio_np = waveform.cpu().numpy().squeeze()
        
        # Compute adaptive threshold
        rms = librosa.feature.rms(y=audio_np, frame_length=2048, hop_length=512)
        threshold = np.mean(rms) + np.std(rms) * 0.5
        
        # Get non-silent intervals
        intervals = librosa.effects.split(
            audio_np,
            top_db=abs(self.config.silence_threshold_db),
            frame_length=2048,
            hop_length=512
        )
        
        # Add padding and merge close intervals
        pad_samples = int(pad_ms * self.config.target_sr / 1000)
        merged_intervals = []
        
        for start, end in intervals:
            start = max(0, start - pad_samples)
            end = min(len(audio_np), end + pad_samples)
            
            if merged_intervals and start <= merged_intervals[-1][1]:
                merged_intervals[-1][1] = end
            else:
                merged_intervals.append([start, end])
        
        # Concatenate non-silent segments
        segments = [audio_np[start:end] for start, end in merged_intervals]
        if not segments:
            return waveform
            
        processed = np.concatenate(segments)
        return torch.from_numpy(processed).to(self.device).unsqueeze(0)

    @torch.no_grad()
    def compute_spectrogram(
        self,
        waveform: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """Compute mel spectrogram with optional normalization.
        
        Args:
            waveform: Input audio tensor
            normalize: Whether to normalize the spectrogram
            
        Returns:
            Mel spectrogram tensor
        """
        try:
            # Cache mel spectrogram transform
            if self._mel_spec_transform is None:
                self._mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.config.target_sr,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    n_mels=self.config.n_mels,
                    normalized=normalize
                ).to(self.device)
            
            # Compute spectrogram
            mel_spec = self._mel_spec_transform(waveform)
            mel_spec = torch.log(mel_spec + 1e-9)
            
            if normalize:
                mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
            
            return mel_spec
            
        except Exception as e:
            raise AudioProcessorError(f"Error computing spectrogram: {str(e)}")

    def get_audio_quality_metrics(self, waveform: torch.Tensor) -> Dict[str, float]:
        """Compute various audio quality metrics.
        
        Args:
            waveform: Input audio tensor
            
        Returns:
            Dictionary of audio quality metrics
        """
        try:
            audio_np = waveform.cpu().numpy().squeeze()
            
            metrics = {
                'peak_level_db': 20 * np.log10(np.max(np.abs(audio_np)) + 1e-9),
                'rms_level_db': 20 * np.log10(np.sqrt(np.mean(audio_np**2)) + 1e-9),
                'loudness_lufs': self.loudness_meter.integrated_loudness(audio_np),
                'dynamic_range_db': 20 * np.log10(np.max(np.abs(audio_np)) / np.mean(np.abs(audio_np)) + 1e-9),
                'zero_crossings_rate': np.mean(np.abs(np.diff(np.signbit(audio_np))))
            }
            
            return metrics
            
        except Exception as e:
            raise AudioProcessorError(f"Error computing audio metrics: {str(e)}")

    def get_prosody_features(
        self,
        waveform: torch.Tensor,
        window_size: Optional[float] = None,
        overlap: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """Extract prosody features from audio using windowed analysis.
        
        Args:
            waveform: Input audio tensor
            window_size: Window size in seconds (default from config)
            overlap: Window overlap ratio (default from config)
            
        Returns:
            Dictionary of prosody features including OpenSmile and wav2vec2 features
        """
        try:
            # Use config values if not specified
            window_size = window_size or self.emotion_config.window_size
            overlap = overlap or self.emotion_config.overlap
            
            # Convert to numpy for prosody analysis
            audio_np = waveform.cpu().numpy().squeeze()
            sr = self.config.target_sr
            
            # Calculate window parameters
            window_samples = int(window_size * sr)
            hop_length = int(window_samples * (1 - overlap))
            
            features = {
                'f0': [],           # Fundamental frequency
                'energy': [],       # Frame energy
                'zcr': [],         # Zero crossing rate
                'rms': [],         # Root mean square energy
                'spectral_centroid': [], # Spectral centroid
                'spectral_flux': [],    # Spectral flux
                'pitch_contour': [],    # Pitch contour
            }
            
            # Process windows
            prev_spec = None  # For spectral flux calculation
            for i in range(0, len(audio_np) - window_samples, hop_length):
                frame = audio_np[i:i + window_samples]
                
                # Basic features
                features['energy'].append(np.sum(frame**2))
                features['zcr'].append(np.mean(np.abs(np.diff(np.signbit(frame)))))
                features['rms'].append(np.sqrt(np.mean(frame**2)))
                
                # Spectral features
                if len(frame) >= 512:
                    spec = np.abs(np.fft.rfft(frame))
                    freqs = np.fft.rfftfreq(len(frame), 1/sr)
                    
                    # Spectral centroid
                    features['spectral_centroid'].append(
                        np.sum(freqs * spec) / (np.sum(spec) + 1e-8)
                    )
                    
                    # Spectral flux
                    if prev_spec is not None:
                        # Calculate difference between consecutive spectra
                        flux = np.sum((spec - prev_spec) ** 2)
                        features['spectral_flux'].append(flux)
                    prev_spec = spec
                    
                    # F0 and pitch contour
                    f0 = self._estimate_f0(frame, sr)
                    features['f0'].append(f0)
                    features['pitch_contour'].append(self._get_pitch_contour(frame, sr))
            
            # Add OpenSmile features if available
            if self.smile is not None:
                try:
                    smile_features = self.smile.process_signal(
                        audio_np,
                        sr
                    )
                    features['opensmile'] = smile_features
                except Exception as e:
                    logger.warning(f"Error computing OpenSmile features: {str(e)}")
            
            # Add wav2vec2 features if available
            if self.wav2vec2_extractor is not None:
                try:
                    inputs = self.wav2vec2_extractor(
                        audio_np,
                        sampling_rate=sr,
                        return_tensors="pt"
                    )
                    features['wav2vec2'] = inputs.input_values.numpy()
                except Exception as e:
                    logger.warning(f"Error computing wav2vec2 features: {str(e)}")
            
            # Convert lists to numpy arrays
            return {k: np.array(v) if isinstance(v, list) else v 
                   for k, v in features.items()}
            
        except Exception as e:
            raise AudioProcessorError(f"Error computing prosody features: {str(e)}")

    def _get_pitch_contour(self, frame: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch contour using autocorrelation."""
        # Window size for pitch estimation (30ms is typical)
        window_size = int(0.03 * sr)
        hop_length = window_size // 2
        
        contour = []
        for i in range(0, len(frame) - window_size, hop_length):
            window = frame[i:i + window_size]
            f0 = self._estimate_f0(window, sr)
            contour.append(f0)
            
        return np.array(contour)

    def process_audio(
        self,
        input_path: str,
        return_spectrogram: bool = False,
        fast_mode: bool = False,
        extract_prosody: bool = False
    ) -> Dict[str, Any]:
        """Complete audio processing pipeline with error handling and memory management.
        
        Args:
            input_path: Path to input audio file
            return_spectrogram: Whether to compute and return spectrogram
            fast_mode: Whether to use faster processing (less accurate)
            extract_prosody: Whether to extract prosody features
            
        Returns:
            Dictionary containing processed audio and metadata
        """
        try:
            logger.info(f"Processing audio file: {input_path}")
            
            # Load and validate
            waveform, metadata = self.load_audio(input_path)
            waveform = self.convert_to_mono(waveform)
            
            # Process audio
            waveform = self.normalize_loudness(waveform, fast_mode=fast_mode)
            waveform = self.remove_silence(waveform)
            
            # Compute metrics
            quality_metrics = self.get_audio_quality_metrics(waveform)
            
            result = {
                'waveform': waveform,
                'metadata': metadata,
                'quality_metrics': quality_metrics
            }
            
            if return_spectrogram:
                result['spectrogram'] = self.compute_spectrogram(waveform)
                
            if extract_prosody:
                result['prosody_features'] = self.get_prosody_features(waveform)
            
            logger.info("Audio processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in audio processing pipeline: {str(e)}")
            raise AudioProcessorError(f"Audio processing failed: {str(e)}")
        finally:
            # Clean up if memory usage is high
            if self._memory_counter > self._max_memory_gb * 0.8:
                self._clear_cache()
