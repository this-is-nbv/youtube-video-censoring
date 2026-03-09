import whisper

whisper_model = whisper.load_model("base")

def extract_transcript(audio_path: str):
    """
    Returns Whisper transcription with word-level timestamps
    """
    return whisper_model.transcribe(
        audio_path,
        word_timestamps=True
    )
