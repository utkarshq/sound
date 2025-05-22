#!/usr/bin/env python3
import sys
import whisper
import json
from pathlib import Path
import numpy as np
import os

def convert_to_serializable(obj):
    """Convert Whisper result to JSON-serializable format"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj

def main():
    """Wrapper for Whisper to handle arguments properly"""
    import argparse
    parser = argparse.ArgumentParser(description="Whisper Transcription Wrapper")
    parser.add_argument("--model", default="large-v3", help="Model to use")
    parser.add_argument("--language", default="ja", help="Language code")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--output_dir", default="/output", help="Output directory")
    parser.add_argument("--output_format", default="json", help="Output format")
    
    args = parser.parse_args()
    
    # Print debug info
    print(f"Debug: Input file path: {args.input}")
    print(f"Debug: Output directory: {args.output_dir}")
    print(f"Debug: Output format: {args.output_format}")

    # Convert input path to Path object for better handling
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load the model
    try:
        model = whisper.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Transcribe
    try:
        result = model.transcribe(
            str(input_path),
            language=args.language,
            verbose=True
        )
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Save output
    # In Docker container, always use the absolute path /output
    output_path = Path("/output")
    print(f"Debug: Using output path: {output_path}")
    
    try:
        # Ensure output directory exists and is writable
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions
        test_file = output_path / "test_write.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()  # Remove test file
            print("Debug: Output directory is writable")
        except Exception as e:
            print(f"Error: Output directory is not writable: {e}", file=sys.stderr)
            sys.exit(1)
        
        if args.output_format == "json":
            # Convert result to JSON-serializable format
            serializable_result = convert_to_serializable(result)
            
            # Add metadata
            serializable_result["metadata"] = {
                "model": args.model,
                "language": args.language,
                "input_file": str(input_path)
            }
            
            # Write JSON output
            json_path = output_path / "transcription.json"
            print(f"Debug: Writing JSON to: {json_path}")
            
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_result, f, ensure_ascii=False, indent=2)
                print(f"Debug: Successfully wrote JSON file to {json_path}")
            except Exception as e:
                print(f"Error writing JSON file: {e}", file=sys.stderr)
                raise
            
            # Write TXT output
            txt_path = output_path / "transcription.txt"
            print(f"Debug: Writing TXT to: {txt_path}")
            
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    for segment in result["segments"]:
                        print(f"[{segment['start']:.3f} --> {segment['end']:.3f}] {segment['text']}", file=f)
                print(f"Debug: Successfully wrote TXT file to {txt_path}")
            except Exception as e:
                print(f"Error writing TXT file: {e}", file=sys.stderr)
                raise
            
        else:
            # Default to txt format
            txt_path = output_path / "transcription.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for segment in result["segments"]:
                    print(f"[{segment['start']:.3f} --> {segment['end']:.3f}] {segment['text']}", file=f)
                    
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        print(f"Error details:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Transcription completed successfully!")

if __name__ == "__main__":
    main()
