# Pipeline configuration
$ErrorActionPreference = "Stop"

function Write-Step {
    param($Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Invoke-PipelineStep {
    param(
        [string]$StepName,
        [scriptblock]$Command
    )
    Write-Step $StepName
    try {
        & $Command
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE"
        }
    }
    catch {
        Write-Host "Error in step '$StepName': $_" -ForegroundColor Red
        exit 1
    }
}

# Ensure model-hub is ready
Invoke-PipelineStep "Checking Models" {
    docker compose run --rm model-hub python /app/scripts/model_manager.py
}

# Extract audio
Invoke-PipelineStep "Extracting Audio" {
    docker compose run --rm ffmpeg -i /input/test.mp4 -vn -acodec pcm_s16le -ar 16000 /output/audio.wav
}

# Run parallel processing
Invoke-PipelineStep "Running Analysis Pipeline" {
    # Transcription
    docker compose run --rm whisper --model large-v3 --language ja --output_dir /output --output_format json /output/audio.wav

    # Speaker Diarization
    docker compose run --rm nemo-diarize python /app/diarize.py /output/audio.wav /output/diarization.json
}

# Align speakers and extract features
Invoke-PipelineStep "Processing Features" {
    docker compose run --rm emotion-features python /app/scripts/align_speakers.py
    docker compose run --rm emotion-features python /app/scripts/extract_wav2vec2.py --input /output/audio.wav --segments /output/aligned.json
    docker compose run --rm emotion-features python /app/scripts/analyze_prosody.py
}

# Process final results
Invoke-PipelineStep "Finalizing Results" {
    docker compose run --rm emotion-features python /app/scripts/process_opensmile.py
}

Write-Host "`nPipeline completed successfully!" -ForegroundColor Green
