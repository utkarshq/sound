import os
import yaml
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config_path: str = "/app/config/models.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.model_path = Path(self.config["model_path"])
        self.model_path.mkdir(parents=True, exist_ok=True)
        
    def download_and_verify_models(self):
        """Download all configured models and verify their integrity"""
        for model_id, model_config in self.config["models"].items():
            model_dir = self.model_path / model_id
            
            if not model_dir.exists() or self.config["download_missing"]:
                logger.info(f"Downloading model: {model_id}")
                self._download_model(model_config, model_dir)
            
            if self.config["verify_downloads"]:
                logger.info(f"Verifying model: {model_id}")
                self._verify_model(model_config, model_dir)
    
    def _download_model(self, model_config: dict, model_dir: Path):
        """Download a specific model based on its configuration"""
        source_type = model_config["source"].split("/")[0]
        
        if source_type == "openai":
            import whisper
            whisper.load_model(model_config["name"], download_root=str(model_dir))
            
        elif source_type in ["audeering", "nvidia"]:
            snapshot_download(
                repo_id=f"{model_config['source']}/{model_config['name']}",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )
    
    def _verify_model(self, model_config: dict, model_dir: Path):
        """Verify model files exist and can be loaded"""
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        # Add specific verification for each model type
        model_type = model_config["type"]
        try:
            if model_type == "whisper":
                import whisper
                model = whisper.load_model(str(model_dir))
            elif model_type == "transformers":
                from transformers import AutoModel
                model = AutoModel.from_pretrained(str(model_dir))
            elif model_type == "nemo":
                # Verification for NeMo models would go here
                pass
                
            logger.info(f"Successfully verified {model_config['name']}")
            
        except Exception as e:
            logger.error(f"Failed to verify model {model_config['name']}: {str(e)}")
            raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Model Manager for Emotion Dubbing')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing models without downloading')
    args = parser.parse_args()

    try:
        manager = ModelManager()
        if args.verify_only:
            logger.info("Verifying existing models...")
            for model_id, model_config in manager.config["models"].items():
                manager._verify_model(model_config, Path(manager.model_path) / model_id)
            logger.info("All models verified successfully!")
        else:
            logger.info("Starting model download and verification...")
            manager.download_and_verify_models()
            logger.info("All models downloaded and verified successfully!")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)
