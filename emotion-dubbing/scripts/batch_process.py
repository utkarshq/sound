#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import logging
import subprocess
from pathlib import Path
import yaml
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, config_path: str = "config/pipeline.yaml"):
        self.config = self._load_config(config_path)
        self.output_base = Path(os.getenv("OUTPUT_DIR", "output"))
        
    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)
    
    def process_file(self, input_file: Path) -> Dict:
        """Process a single input file through the pipeline"""
        # Create unique output directory for this file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"{input_file.stem}_{timestamp}"
        output_dir = self.output_base / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing {input_file} -> {output_dir}")

        try:
            # Run the pipeline with specific input/output directories
            env = os.environ.copy()
            env.update({
                "INPUT_DIR": str(input_file.parent),
                "OUTPUT_DIR": str(output_dir),
                "INPUT_FILE": input_file.name
            })

            cmd = ["docker", "compose", "run", "--rm"]
            for service in ["ffmpeg", "whisper", "nemo-diarize", "emotion-features"]:
                cmd.extend(["-e", f"INPUT_DIR={env['INPUT_DIR']}", 
                          "-e", f"OUTPUT_DIR={env['OUTPUT_DIR']}", 
                          "-e", f"INPUT_FILE={env['INPUT_FILE']}"])
            
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                capture_output=True,
                text=True
            )

            # Create status file
            status = {
                "job_id": job_id,
                "input_file": str(input_file),
                "output_dir": str(output_dir),
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }

            with open(output_dir / "status.json", "w") as f:
                json.dump(status, f, indent=2)

            return status

        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline failed for {input_file}: {e}")
            status = {
                "job_id": job_id,
                "input_file": str(input_file),
                "output_dir": str(output_dir),
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            with open(output_dir / "status.json", "w") as f:
                json.dump(status, f, indent=2)
            return status

    def process_batch(self, input_pattern: str, max_concurrent: int = 2) -> List[Dict]:
        """Process multiple files concurrently"""
        input_files = [Path(f) for f in glob.glob(input_pattern)]
        if not input_files:
            logger.error(f"No files found matching pattern: {input_pattern}")
            return []

        logger.info(f"Found {len(input_files)} files to process")
        results = []

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_file = {executor.submit(self.process_file, f): f for f in input_files}
            for future in future_to_file:
                file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed processing {file}")
                except Exception as e:
                    logger.error(f"Failed to process {file}: {e}")

        return results

def main():
    parser = argparse.ArgumentParser(description="Batch process videos for emotion dubbing")
    parser.add_argument("input_pattern", help="Glob pattern for input files (e.g., 'input/*.mp4')")
    parser.add_argument("--max-concurrent", type=int, default=2, 
                      help="Maximum number of files to process concurrently")
    parser.add_argument("--output-dir", help="Base output directory")
    args = parser.parse_args()

    if args.output_dir:
        os.environ["OUTPUT_DIR"] = args.output_dir

    processor = BatchProcessor()
    results = processor.process_batch(args.input_pattern, args.max_concurrent)

    # Write summary report
    summary = {
        "total_files": len(results),
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results
    }

    with open("batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Batch processing complete. {summary['completed']}/{summary['total_files']} files processed successfully.")

if __name__ == "__main__":
    main()
