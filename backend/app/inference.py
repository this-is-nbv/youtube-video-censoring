import torch
from transformers import BertTokenizer, BertModel
from detoxify import Detoxify

from app.model.mfustsvd_ta import MFusTSVD_TA
from app.audio import extract_hlla, extract_dhla, combine_audio_features
from app.transcript import extract_transcript
from app.youtube.downloader import download_audio

LABELS = ["SAFE", "VIOLENCE", "SEXUAL"]

# -------------------------------
# Load models once
# -------------------------------

model = MFusTSVD_TA()
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

toxicity_model = Detoxify("original")

# -------------------------------
# Always censor words (kids safe)
# -------------------------------

ALWAYS_CENSOR = {
    "fuck","fucking","fucker",
    "shit","shitty",
    "bitch",
    "porn",
    "sex",
    "sexual",
    "pussy",
    "dick",
    "asshole",
    "jerk"
}

# -------------------------------
# Context-based offensive roots
# -------------------------------

CONTEXT_ROOTS = [
    "kill","murder","shoot","stab","attack",
    "choke","strangl","beat","punch",
    "idiot","moron","racist","slut","whore",
    "hate","loser","stupid","dumb","jerk"
]

def is_offensive(word):

    word = word.lower()

    if any(bad in word for bad in ALWAYS_CENSOR):
        return True

    if any(root in word for root in CONTEXT_ROOTS):
        return True

    return False


# -------------------------------
# Extract word timestamps
# -------------------------------

def extract_word_timings(result):

    words = []

    for seg in result.get("segments", []):
        for w in seg.get("words", []):

            clean_word = w["word"].lower().strip(".,!?")

            words.append({
                "word": clean_word,
                "start": round(w["start"], 2),
                "end": round(w["end"], 2)
            })

    return words


# -------------------------------
# Build sentence windows
# -------------------------------

def build_sentences(transcript):

    sentences = []

    for seg in transcript.get("segments", []):

        sentences.append({
            "text": seg["text"].strip(),
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2)
        })

    return sentences


# -------------------------------
# Context-aware profanity detection
# -------------------------------

def detect_profanity(sentences, words):

    profanity_windows = []
    profane_words = set()
    seen = set()

    # --------------------------
    # 1️⃣ Always censor explicit words
    # --------------------------

    for w in words:

        clean = w["word"]

        if is_offensive(clean):

            key = (w["start"], w["end"])

            if key not in seen:
                profanity_windows.append({
                    "start": w["start"],
                    "end": w["end"]
                })
                seen.add(key)

            profane_words.add(clean)

    # --------------------------
    # 2️⃣ NLP context detection
    # --------------------------

    for s in sentences:

        scores = toxicity_model.predict(s["text"])

        toxicity = scores["toxicity"]
        obscene = scores["obscene"]
        threat = scores["threat"]
        insult = scores["insult"]
        identity_attack = scores["identity_attack"]
        severe = scores["severe_toxicity"]

        if (
            toxicity > 0.60 or
            obscene > 0.45 or
            threat > 0.50 or
            insult > 0.60 or
            identity_attack > 0.45 or
            severe > 0.40
        ):

            for w in words:

                if s["start"] <= w["start"] <= s["end"]:

                    clean = w["word"]

                    if is_offensive(clean):

                        key = (w["start"], w["end"])

                        if key not in seen:
                            profanity_windows.append({
                                "start": w["start"],
                                "end": w["end"]
                            })
                            seen.add(key)

                        profane_words.add(clean)

    return profanity_windows, list(profane_words)


# -------------------------------
# Main inference pipeline
# -------------------------------

def run_inference(video_id: str):

    # 1️⃣ Download audio
    audio_path = download_audio(video_id)

    # 2️⃣ Whisper transcription
    transcript = extract_transcript(audio_path)

    # 3️⃣ Extract word timestamps
    words = extract_word_timings(transcript)

    # 4️⃣ Sentence windows
    sentences = build_sentences(transcript)

    # 5️⃣ Profanity detection
    profanity_windows, profane_words = detect_profanity(sentences, words)

    # 6️⃣ Text features for MFusTSVD
    text = transcript["text"]

    tokens = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        text_feat = bert(**tokens).last_hidden_state

    # 7️⃣ Audio features
    hlla = extract_hlla(audio_path)
    dhla = extract_dhla(audio_path)
    audio_feat = combine_audio_features(hlla, dhla)

    # 8️⃣ MFusTSVD inference
    with torch.no_grad():
        probs = model(text_feat, audio_feat)

    label = LABELS[probs.argmax(dim=1).item()]

    print(f"🔍 MFusTSVD-TA Prediction for {video_id}: {label}")
    print("🚫 Profanity windows:", profanity_windows)
    print("🚫 Profane words:", profane_words)

    return {
        "label": label,
        "profanity_windows": profanity_windows,
        "profane_words": profane_words
    }