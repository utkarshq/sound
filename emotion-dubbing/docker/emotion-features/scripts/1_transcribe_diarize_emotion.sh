#!/bin/bash

# Step 1: Extract audio
docker compose run --rm ffmpeg -i /input/test.mp4 -vn /output/audio.wav

# Step 2: Transcribe
docker compose run --rm whisper whisper --model large-v3 --language ja /output/audio.wav --output_dir /output --output_format json

# Step 3: Diarize
docker compose run --rm nemo-diarize python /app/diarize.py /output/audio.wav /output/diarization.json

# Step 4: Align speakers
docker compose run --rm emotion-features python /app/scripts/align_speakers.py

# Step 5: Extract emotion features
docker compose run --rm emotion-features python /app/scripts/extract_wav2vec2.py --input /output/audio.wav --segments /output/aligned.json
docker compose run --rm emotion-features SMILExtract -C /opensmile/config/emobase.conf -I /output/audio.wav -O /output/opensmile.csv
docker compose run --rm emotion-features python /app/scripts/process_opensmile.py

# Step 6: Combine features
docker compose run --rm emotion-features python -c "
import json;
with open('/output/wav2vec2_emotion.json') as f: w2v = json.load(f)['emotion_wav2vec2'];
with open('/output/opensmile.json') as f: opensmile = json.load(f)['opensmile_features'];
with open('/output/aligned.json') as f: aligned = json.load(f);
for seg, w, o in zip(aligned, w2v, opensmile): seg['emotion_vector'] = w + o;
with open('/output/final_output.json', 'w') as f: json.dump(aligned, f, indent=2);
"