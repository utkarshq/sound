import torchaudio  
from pydub import AudioSegment  
import json  

def extract_emotion(wav_path, segments):  
    # Load audio  
    audio = AudioSegment.from_wav(wav_path)  

    # Load model  
    model = Wav2Vec2ForSequenceClassification.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")  
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")  

    # Process each segment  
    emotion_vectors = []  
    for seg in segments:  
        start_ms = int(seg["start"] * 1000)  
        end_ms = int(seg["end"] * 1000)  
        segment = audio[start_ms:end_ms]  
        segment.export("temp.wav", format="wav")  

        # Extract features  
        waveform, sr = torchaudio.load("temp.wav")  
        inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt")  
        embeddings = model(**inputs).logits.mean(dim=1).tolist()  
        emotion_vectors.append(embeddings[0])  

    # Save per-segment vectors  
    with open("wav2vec2_emotion.json", "w") as f:  
        json.dump({"emotion_wav2vec2": emotion_vectors}, f)  

# Usage: Pass aligned.json and audio.wav  
with open("aligned.json") as f:  
    segments = json.load(f)  
extract_emotion("audio.wav", segments)  