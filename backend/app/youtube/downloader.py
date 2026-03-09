import subprocess
from pathlib import Path

# downloader.py → youtube → app → backend
BACKEND_DIR = Path(__file__).resolve().parents[2]

AUDIO_DIR = BACKEND_DIR / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def download_audio(video_id: str) -> str:
    """
    Downloads YouTube audio and converts to WAV
    """
    output_path = AUDIO_DIR / f"{video_id}.wav"
    print(str(output_path))
    url = f"https://www.youtube.com/watch?v={video_id}"

    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path),
        url
    ]

    subprocess.run(command, check=True)
    return str(output_path)
