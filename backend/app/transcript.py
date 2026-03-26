from faster_whisper import WhisperModel
import torch

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Faster Whisper running on:", device)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model_size = "small"

whisper_model = WhisperModel(
    model_size,
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)

# --------------------------------------------------
# TRANSCRIPTION
# --------------------------------------------------

def extract_transcript(audio_path: str):

    segments, info = whisper_model.transcribe(
        audio_path,
        word_timestamps=True
    )

    result = {
        "text": "",
        "segments": []
    }

    for seg in segments:

        words = []

        if seg.words:

            for w in seg.words:

                words.append({
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end)
                })

        result["segments"].append({
            "text": seg.text,
            "start": float(seg.start),
            "end": float(seg.end),
            "words": words
        })

        result["text"] += seg.text + " "

    return result