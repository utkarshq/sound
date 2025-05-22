# Audio Processing Module

This module provides advanced audio processing capabilities for the emotion dubbing pipeline, including:

- High-quality audio loading and resampling
- Intelligent stereo to mono conversion
- Loudness normalization (EBU R128)
- Adaptive silence removal
- GPU-accelerated spectrogram computation

## Usage

```python
from src.audio_processing import AudioProcessor, AudioConfig

# Create configuration
config = AudioConfig(
    target_sr=16000,          # Target sample rate
    target_loudness=-23.0,    # Target loudness (LUFS)
    silence_threshold_db=-40,  # Silence threshold
    use_gpu=True             # Enable GPU acceleration
)

# Initialize processor
processor = AudioProcessor(config)

# Process audio file
result = processor.process_audio(
    "input.wav",
    return_spectrogram=True  # Also compute mel spectrogram
)

# Access results
waveform = result['waveform']            # Processed audio
metadata = result['metadata']            # Audio metadata
spectrogram = result['spectrogram']      # Mel spectrogram

# Access individual components
waveform, metadata = processor.load_audio("input.wav")
mono = processor.convert_to_mono(waveform)
normalized = processor.normalize_loudness(mono)
processed = processor.remove_silence(normalized)
spec = processor.compute_spectrogram(processed)
```

## Features

### Audio Loading and Resampling
- Efficient audio file loading with torchaudio
- Cached resamplers for common sample rates
- Automatic resampling to target sample rate

### Stereo to Mono Conversion
- Intelligent channel mixing
- Weighted average for better mono conversion
- Preserves spatial information

### Loudness Normalization
- EBU R128 compliant loudness normalization
- Integrated LUFS measurement
- Soft-knee peak limiting
- Preserves dynamic range

### Silence Removal
- Adaptive threshold based on signal statistics
- Intelligent padding around speech segments
- Merge close segments
- Prevents choppy output

### Spectrogram Computation
- GPU-accelerated mel spectrogram computation
- Optional normalization
- Configurable parameters
  - Number of mel bands
  - FFT window size
  - Hop length

### GPU Support
- Automatic GPU detection and usage
- Efficient memory management
- Cached transforms for better performance

## Configuration

The `AudioConfig` class provides the following configuration options:

```python
@dataclass
class AudioConfig:
    target_sr: int = 16000          # Target sample rate
    target_loudness: float = -23.0   # Target loudness in LUFS
    silence_threshold_db: float = -40.0  # Silence threshold
    min_segment_dur: float = 0.1     # Minimum segment duration
    n_mels: int = 80                # Number of mel bands
    n_fft: int = 2048              # FFT window size
    hop_length: int = 512           # Hop length
    use_gpu: bool = True           # Use GPU if available
```

## Error Handling

The module provides comprehensive error handling:

- Graceful fallback to CPU when GPU is unavailable
- Informative error messages
- Extensive logging
- Input validation

## Performance Optimization

- Cached resamplers for common sample rates
- GPU acceleration where beneficial
- Efficient memory usage
- Vectorized operations

## Requirements

- torch
- torchaudio
- librosa
- numpy
- pyloudnorm
- future

## Testing

Run the test suite:

```bash
pytest tests/test_audio_processing.py
```

This includes:
- Unit tests for individual components
- Integration tests for full pipeline
- GPU tests when available
- Edge case handling
