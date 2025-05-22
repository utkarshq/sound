import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification
import opensmile
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class FusionLayer(nn.Module):
    def __init__(self, frame_dim=384, embedding_dim=1024):
        super().__init__()
        self.frame_projection = nn.Linear(frame_dim, embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, frame_features, embeddings):
        # Project frame features to embedding dimension
        frame_proj = self.frame_projection(frame_features)
        
        # Apply layer normalization
        frame_proj = self.layer_norm(frame_proj)
        
        # Self-attention over sequence with dropout
        attn_output, _ = self.attention(frame_proj, frame_proj, frame_proj)
        attn_output = self.dropout(attn_output)
        
        # Concatenate and fuse
        fused = torch.cat([attn_output, embeddings], dim=-1)
        return self.fusion(fused)

class EmotionAnalyzer:
    def __init__(self, config_path: str = None):
        logger.info("Initializing EmotionAnalyzer")
        
        # Optimize CUDA settings
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize OpenSMILE with enhanced feature set
        self.opensmile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors_100Hz
        )
        
        # Initialize wav2vec2 with optimized settings
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            "audeering/wav2vec2-large-robust",
            torchscript=True  # Enable TorchScript optimization
        ).to(self.device).eval()  # Set to eval mode for inference
        
        # Initialize enhanced fusion layer
        self.fusion_layer = FusionLayer().to(self.device)
        
        # Cache clearing
        torch.cuda.empty_cache()
        
        logger.info(f"EmotionAnalyzer initialized on device: {self.device}")

    def extract_opensmile_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract frame-level features using OpenSMILE with optimized processing"""
        logger.info("Extracting OpenSMILE features")
        
        # Process in chunks for memory efficiency
        chunk_size = 160000  # 10 seconds at 16kHz
        features_list = []
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            features = self.opensmile.process_signal(
                chunk,
                sampling_rate=16000
            )
            features_list.append(features.values)
        
        features = np.concatenate(features_list, axis=0)
        return torch.FloatTensor(features).to(self.device)

    def extract_wav2vec2_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract utterance-level embeddings using wav2vec2 with optimized batch processing"""
        logger.info("Extracting wav2vec2 embeddings")
        
        with torch.cuda.amp.autocast(), torch.no_grad():
            outputs = self.wav2vec2(audio.to(self.device))
            return outputs.last_hidden_state

    def temporal_align(
        self, 
        frame_features: torch.Tensor,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align frame-level and utterance-level features temporally with enhanced precision"""
        logger.info("Performing temporal alignment")
        
        # Use bilinear interpolation for better quality
        embeddings = nn.functional.interpolate(
            embeddings.unsqueeze(0).transpose(1, 2),
            size=frame_features.shape[0],
            mode='linear',
            align_corners=True
        ).transpose(1, 2).squeeze(0)
        
        # Apply Gaussian smoothing to reduce interpolation artifacts
        gaussian_kernel = torch.tensor(
            [0.054, 0.244, 0.403, 0.244, 0.054],
            device=self.device
        ).view(1, 1, -1)
        
        embeddings = nn.functional.pad(
            embeddings.unsqueeze(1),
            (2, 2),
            mode='replicate'
        )
        embeddings = nn.functional.conv1d(
            embeddings,
            gaussian_kernel.expand(embeddings.size(2), -1, -1),
            groups=embeddings.size(2)
        ).squeeze(1)
        
        return frame_features, embeddings

    @torch.compile  # Enable TorchScript compilation for faster inference
    def process(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """Complete emotion analysis pipeline with optimized processing"""
        logger.info(f"Processing audio file: {audio_path}")

        # Load and preprocess audio
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)

        # Extract features with automatic mixed precision
        with torch.cuda.amp.autocast():
            smile_features = self.extract_opensmile_features(audio.numpy())
            w2v_embeddings = self.extract_wav2vec2_embeddings(audio)
            
            # Align features temporally
            aligned_features, aligned_embeddings = self.temporal_align(
                smile_features, w2v_embeddings
            )
            
            # Fuse features
            fused_features = self.fusion_layer(aligned_features, aligned_embeddings)

        return {
            'frame_features': smile_features.cpu(),
            'embeddings': w2v_embeddings.cpu(),
            'aligned_features': aligned_features.cpu(),
            'fused_features': fused_features.cpu()
        }
