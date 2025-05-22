from nemo.collections.asr.models import ClusteringDiarizer
import sys, json, os
from omegaconf import OmegaConf
import torch

def optimize_memory():
    """Optimize GPU memory usage"""
    import torch
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()

def select_optimal_model(num_speakers: int) -> str:
    """Select optimal diarization model based on scenario"""
    if num_speakers <= 2:
        return "titanet_large"  # Best for 1-2 speakers
    elif num_speakers <= 5:
        return "ecapa_tdnn"     # Best for 2-5 speakers
    else:
        return "titanet_large"  # Best for >5 speakers

def diarize(input_audio: str, output_json: str, min_speakers: int = 1, max_speakers: int = 8):
    """Run speaker diarization using NeMo with GPU optimization"""
    logger.info(f"Starting diarization for {input_audio}")
    
    # Optimize GPU memory
    optimize_memory()
    
    # Select model based on expected speakers
    model_name = select_optimal_model((min_speakers + max_speakers) // 2)
    logger.info(f"Selected model: {model_name}")
    
    # Create optimized config for diarizer
    diarizer_config = {
        'manifest_filepath': input_audio,
        'out_dir': str(Path(output_json).parent),
        'diarizer.manifest_filepath': input_audio,
        'diarizer.out_dir': str(Path(output_json).parent),
        'diarizer.speaker_embeddings.model_path': model_name,
        'diarizer.clustering.parameters.oracle_num_speakers': False,
        'diarizer.clustering.parameters.min_speakers': min_speakers,
        'diarizer.clustering.parameters.max_speakers': max_speakers,
        'diarizer.msdd_model.model_path': 'diar_msdd_telephonic',
        'diarizer.oracle_vad': False,
        'diarizer.vad.model_path': 'vad_multilingual_marblenet',
        'diarizer.vad.parameters.onset': 0.8,
        'diarizer.vad.parameters.offset': 0.6,
        'diarizer.clustering.parameters.oracle_num_speakers': False,
        'diarizer.clustering.parameters.similarity_threshold': 0.7
    cfg = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": "none",
            "out_dir": "/output/diarizer",
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "window_length_in_sec": 1.5,
                "shift_length_in_sec": 0.75,
                "embeddings_in_memory": True
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": False,
                    "max_num_speakers": 8,
                    "min_num_speakers": 1,
                    "spectral_clustering_n_iter": 1000
                }
            },
            "msdd_model": {
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "onset": 0.5,
                    "offset": 0.5
                }
            }
        }
    })
    
    # Set up GPU if available
    if torch.cuda.is_available():
        cfg.diarizer.cuda = True
    
    # Initialize diarizer with config
    diarizer = ClusteringDiarizer(cfg=cfg)
    
    # Verify audio file exists
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Audio file {input_audio} not found")
    
    # Run diarization
    diarizer.diarize(paths2audio_files=[input_audio])
    
    # Write results
    rttm_file = os.path.join(cfg.diarizer.out_dir, 'pred_rttms', os.path.basename(input_audio).replace('.wav', '.rttm'))
    if os.path.exists(rttm_file):
        # Convert RTTM to JSON
        segments = []
        with open(rttm_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                segments.append({
                    "start": float(parts[3]),
                    "end": float(parts[3]) + float(parts[4]),
                    "speaker_id": parts[7]
                })
        
        # Write JSON output
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(segments, f, indent=2)
    else:
        raise FileNotFoundError(f"RTTM file not found at {rttm_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python diarize.py <input_audio> <output_json>")
        sys.exit(1)
    
    try:
        diarize(sys.argv[1], sys.argv[2])
    except Exception as e:
        print(f"Error during diarization: {e}", file=sys.stderr)
        sys.exit(1)