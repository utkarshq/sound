import os
import json

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

# Check inputs
check_file("/output/audio.json")
check_file("/output/diarization.json")


# Load Whisper transcript and NeMo diarization
with open("/output/audio.json") as f:  # Note /output/ prefix
    transcript = json.load(f)["segments"]
with open("/output/diarization.json") as f:
    diarization = json.load(f)

# Assign speaker IDs to transcript segments

def align_speakers(transcript, diarization):
    aligned = []
    for seg in transcript:
        max_overlap = 0
        best_speaker = "unknown"
        for dia in diarization:
            overlap_start = max(seg["start"], dia["start"])
            overlap_end = min(seg["end"], dia["end"])
            overlap = overlap_end - overlap_start
            if overlap > max_overlap and overlap > 0:  # Threshold: >0 sec overlap
                max_overlap = overlap
                best_speaker = dia["speaker_id"]
        seg["speaker_id"] = best_speaker
        aligned.append(seg)
    return aligned

with open("aligned.json", "w") as f:
    json.dump(aligned, f, ensure_ascii=False)