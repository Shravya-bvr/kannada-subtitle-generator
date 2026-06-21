import subprocess
import os

VIDEO_PATH = "input/video.mp4"
AUDIO_PATH = "audio/extracted.wav"

os.makedirs("audio", exist_ok=True)

print("🎧 Extracting audio from video...")

cmd = [
    "ffmpeg", "-y",
    "-i", VIDEO_PATH,
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-acodec", "pcm_s16le",
    AUDIO_PATH
]

subprocess.run(cmd, check=True)

print("✅ Audio extracted successfully!")
print("📄 Saved at:", AUDIO_PATH)
