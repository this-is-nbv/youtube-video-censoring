import torch
from transformers import BertTokenizer, BertModel
from detoxify import Detoxify

from app.model.mfustsvd_ta import MFusTSVD_TA
from app.audio import extract_hlla, extract_dhla, combine_audio_features
from app.transcript import extract_transcript
from app.youtube.downloader import download_audio

# --------------------------------------------------
# DEVICE CONFIGURATION
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Running on:", device)

LABELS = ["SAFE", "VIOLENCE", "SEXUAL"]

# --------------------------------------------------
# LOAD MODELS (ONCE)
# --------------------------------------------------

model = MFusTSVD_TA().to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

bert = BertModel.from_pretrained(
    "bert-base-uncased"
).to(device)

bert.eval()

toxicity_model = Detoxify(
    "original",
    device=device
)

# --------------------------------------------------
# ALWAYS CENSOR WORDS
# --------------------------------------------------

ALWAYS_CENSOR = {
    "fuck", "fucking", "fucker",
    "shit", "shitty",
    "bitch",
    "porn",
    "sex",
    "sexual",
    "pussy",
    "dick",
    "asshole",
    "jerk"
}

# --------------------------------------------------
# CONTEXT ROOTS
# --------------------------------------------------

CONTEXT_ROOTS = [
    "kill", "murder", "shoot", "attack",
    "choke", "strangl", "beat", "punch",
    "idiot", "moron", "racist", "slut", "whore",
    "hate", "loser", "stupid", "dumb", "jerk"
]

# --------------------------------------------------
# OFFENSIVE WORD CHECK
# --------------------------------------------------

def is_offensive(word):

    word = word.lower()

    if any(bad in word for bad in ALWAYS_CENSOR):
        return True

    if any(root in word for root in CONTEXT_ROOTS):
        return True

    return False

# --------------------------------------------------
# WORD TIMESTAMPS
# --------------------------------------------------

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

# --------------------------------------------------
# SENTENCE WINDOWS
# --------------------------------------------------

def build_sentences(transcript):

    sentences = []

    for seg in transcript.get("segments", []):

        sentences.append({
            "text": seg["text"].strip(),
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2)
        })

    return sentences

# --------------------------------------------------
# PROFANITY DETECTION
# --------------------------------------------------

def detect_profanity(sentences, words):

    profanity_windows = []
    profane_words = set()
    seen = set()

    # Always censor explicit words

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

    # Context detection (NLP)

    for s in sentences:

        scores = toxicity_model.predict(
            s["text"]
        )

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

                if (
                    s["start"]
                    <= w["start"]
                    <= s["end"]
                ):

                    clean = w["word"]

                    if is_offensive(clean):

                        key = (
                            w["start"],
                            w["end"]
                        )

                        if key not in seen:

                            profanity_windows.append({
                                "start": w["start"],
                                "end": w["end"]
                            })

                            seen.add(key)

                        profane_words.add(clean)

    return profanity_windows, list(profane_words)

# --------------------------------------------------
# MAIN PIPELINE (SEQUENTIAL VERSION)
# --------------------------------------------------

def run_inference(video_id: str):

    print("Starting analysis:", video_id)

    # 1 Download audio

    audio_path = download_audio(video_id)

    # 2 Transcription

    transcript = extract_transcript(
        audio_path
    )

    # 3 Words

    words = extract_word_timings(
        transcript
    )

    # 4 Sentences

    sentences = build_sentences(
        transcript
    )

    # 5 Profanity detection

    profanity_windows, profane_words = detect_profanity(
        sentences,
        words
    )

    # 6 TEXT FEATURES

    text = transcript["text"]

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    tokens = {
        k: v.to(device)
        for k, v in tokens.items()
    }

    with torch.no_grad():

        text_feat = bert(
            **tokens
        ).last_hidden_state

    # 7 AUDIO FEATURES

    hlla = extract_hlla(
        audio_path
    )

    dhla = extract_dhla(
        audio_path
    )

    audio_feat = combine_audio_features(
        hlla,
        dhla
    ).to(device)

    # 8 MODEL INFERENCE

    with torch.no_grad():

        probs = model(
            text_feat,
            audio_feat
        )

    label = LABELS[
        probs.argmax(dim=1).item()
    ]

    print("Prediction:", label)
    print("Profanity windows:", profanity_windows)

    return {
        "label": label,
        "profanity_windows": profanity_windows,
        "profane_words": profane_words
    }